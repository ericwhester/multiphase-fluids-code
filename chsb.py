"""Cahn-Hilliard Stokes flow""" 
import dedalus.public as de
import pandas as pd
import numpy as np
from dedalus.tools import post
from dedalus.extras import flow_tools
import file_tools as flt
def mag(x): return np.log10(np.abs(x)+1e-16)
import glob
from mpi4py import MPI
import time
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
import matplotlib.pyplot as plt
import os
import logging
root = logging.root
for h in root.handlers: h.setLevel("INFO") 
logger = logging.getLogger(__name__)


from math import floor, log10, ceil
multiples = {2:log10(2), 5:log10(5)}

def calc_mult(x):
    if x < multiples[2]: return 1
    elif x < multiples[5]: return 2
    elif x < 1: return 5
    else: return False

def round_to_power(x):
    power, mult = floor(log10(x)), log10(x)%1
    return 10**power * calc_mult(mult)

def read_file(name):
    with open(name,mode='r',encoding='utf-16') as f:
        return f.readlines()[0]

def cast(op):
    try:    return op.evaluate()['g']
    except: pass
    try:    return op.evaluate().value
    except: pass
    try:    return op.value
    except: pass
    try:    return op.evaluate().data
    except: pass
    return op

def calc_noise_coeffs(domain,params,ncoeffs=96):
    global_coeff_shape = domain.dist.coeff_layout.global_shape(domain.dealias)
    coeffs = np.zeros(global_coeff_shape,dtype=np.complex128)
    np.random.seed(2000)
    pert_shape = (ncoeffs,)*len(params['dims'])
    perts = 2*(np.random.random(pert_shape) - .5) + 2j*(np.random.random(pert_shape) - .5)
    mags = sum(-k**2 for k in domain.all_elements())
    coeffs[tuple(slice(0,ncoeffs) for _ in range(len(params['dims'])))] = perts
    local_slice = domain.dist.coeff_layout.slices(domain.dealias)
    subarr = coeffs[local_slice]*params['Δc']*np.exp(mags/15**2)
    return subarr

def initial_condition(domain, params, fields, ncoeffs=96):
    r = sum(g**2 for g in domain.grids(domain.dealias))**(1/2)
    ε, c10, Δc, Lx = [params[a] for a in ['ε','c10','Δc','Lx']]
    inside = 0.5*(1 + np.tanh(-(r-.7*Lx)/(2**(1/2)*ε)))
    np.random.seed(2000)
    # random noise on 64x64x128 grid
    target_scales = {'nx':64,'ny':64,'nz':128}
    scaling = tuple(target_scales[f'n{d}']/params[f'n{d}'] for d in params['dims'])
    target_size = [target_scales[f'n{d}'] for d in params['dims']]
    noisef = domain.new_field()
    noisef.set_scales(scaling)
    local_slice = domain.dist.grid_layout.slices(scaling)
    noise = np.random.random(target_size)
    subnoise = noise[local_slice] - .5
    noisef['g'] = subnoise
    noisef.set_scales(domain.dealias)
    fields[0]['g'] = inside*(c10 + Δc*noisef['g'])
    fields[1]['g'] = inside*((1 - c10) - Δc*noisef['g'])
    return 

def step_solvers(dt,solvers,stores,flow=True):
    solvers[0].step(dt)
    for f in solvers[0].problem.variables: stores[f]['g'] = solvers[0].state[f]['g']
    if flow:
        solvers[1].solve()
        for f in solvers[1].problem.variables: stores[f]['g'] = solvers[1].state[f]['g']

def run_chs(params,return_solver=False):
    # load parameters
    dims = params['dims']
    dl = dims[-1]
    ns = list(range(1,4))
    flow = params['flow']

    # set up domain 
    basis = {d: de.Fourier if d != dims[-1] else de.Chebyshev for d in dims}
    bases = [basis[d](d, params[f'n{d}'], interval=(-params["Lx"],params["Lx"]),dealias=params['dealias']) for d in dims]
    domain = de.Domain(bases, grid_dtype=np.float64,mesh=params.get('pmesh'))

    # define variables
    c_vars = [f'{f}{i}{g}' for f in 'cm' for i in '12' for g in ['',dl]] + ['lapc1','lapc2']
    u_vars = ['p'] + [f'u{xk}' for xk in dims] + [f'u{xk}{dl}' for xk in dims]
    stores = {n:domain.new_field() for n in c_vars + u_vars}

    # parameters for each problem
    params_c = ['ε','σ12','σ13','σ23','flow']
    params_u = ['ε','σ12','σ13','σ23','Bo','We','Ca'] + [f'ρ{i}' for i in ns]

    # # connecting fields between each problem
    # stores_c = [f'u{xk}' for xk in dims] + ['p'] # velocities for advection, and saving everything
    # stores_u = [f'c{i}{g}' for i in [1,2] for g in ['',dl]] + [f'lapc{i}' for i in [1,2]]

    # define substitutions and equations for each problem
    subs_c, subs_u = [], []
    eqs_c, eqs_u = [], []

    # c substitutions
    subs_c.append(('c3','1-c1-c2'))
    for i in ns:
        for dim in dims[:-1]: subs_c.append((f'c{i}{dim}', f'd{dim}(c{i})'))
    for dim in dims: subs_c.append((f'c3{dim}', f'-(c1{dim} + c2{dim})'))
    subs_c.append((f'lapc3',f'-(lapc1+lapc2)'))    
    for i in ns:
        subs_c.append((f'σ{i}{i}','0'))
        for j in range(i+1,4):
            subs_c.append((f'σ{j}{i}',f'σ{i}{j}'))
    subs_c.append(('χ1', 'σ12 + σ13 - σ23'))
    subs_c.append(('χ2', 'σ12 - σ13 + σ23'))
    subs_c.append(('χ3', '-σ12 + σ13 + σ23'))
    subs_c.append(('g(c)', 'c**2 * (1-c)**2'))
    subs_c.append(('dg(c)', '2*c*(1-c)*(1-2*c)'))
    subs_c.append(('hlap(c)', '+'.join([f'd{xk}(d{xk}(c))' for xk in dims[:-1]])))
    subs_c.append(('lap(c,cd)', '+'.join([f'd{xk}(d{xk}(c))' for xk in dims[:-1]]+[f'd{dl}(cd)'])))
    subs_c.append(('q(c)', '-6*c**2 + 4*c**3'))
    subs_c.append(('C1', 'integ(c1)'))
    subs_c.append(('C2', 'integ(c2)'))
    subs_c.append(('C3', 'integ(c3)'))
    subs_c.append(('f1', '({})'.format("+".join([f'χ{i}*g(c{i})' for i in ns]))))
    subs_c.append((f'graddotgrad(ci,ci{dl},cj,cj{dl})', ' + '.join([f'd{xk}(ci)*d{xk}(cj)' for xk in dims[:-1]] + [f'ci{dl}*cj{dl}'])))
    subs_c.append(('f2', '-(1/2)*ϵ**2*({})'.format('+'.join([f'σ{i}{j}*graddotgrad(c{i},c{i}{dl},c{j},c{j}{dl})' for i in ns for j in ns if j != i]) )))
    subs_c.append(('W', 'f1 + f2'))
    subs_c.append(('E', 'integ(W)'))
    for i in ns:
        subs_c.append((f'μ{i}_op',f'χ{i}*dg(c{i})'+'+ε**2*({})'.format('+'.join([f'σ{i}{j}*lapc{j}' for j in ns if j != i]))))
    subs_c.append(('m1_lhs', f'((4*χ1+2*χ3)*c1 + (2*χ3-2*χ2)*c2) + ε**2*((-σ12-3*σ13+σ23)*lapc1 + 2*(σ12-σ13)*lapc2)'))
    subs_c.append(('m2_lhs', f'((2*χ3-2*χ1)*c1 + (4*χ2+2*χ3)*c2) + ε**2*(2*(σ12-σ23)*lapc1 + (-σ12+σ13-3*σ23)*lapc2)'))
    subs_c.append(('m1_rhs', f'(2*χ1*q(c1) -   χ2*q(c2) - χ3*q(c3))'))
    subs_c.append(('m2_rhs', f'(- χ1*q(c1) + 2*χ2*q(c2) - χ3*q(c3))'))
    for i in [1,2]:
        subs_c.append((f'm{i}_op', f'm{i}_lhs + m{i}_rhs'))
        subs_c.append((f'lapc{i}_op', f'lap(c{i},c{i}{dl})'))
    subs_c.append((f'adv(c,c{dl})', '+'.join([f'u{xk}*d{xk}(c)' for xk in dims[:-1]] + [f'u{dl}*c{dl}'])))
    subs_c.append(('dtc1', f'lap(m1,m1{dl}) - adv(c1,c1{dl})'))
    subs_c.append(('dtc2', f'lap(m2,m2{dl}) - adv(c2,c2{dl})'))

    # c equations
    for i in [1,2]:
        for f in 'cm':
            eqs_c.append(f'd{dl}({f}{i}) - {f}{i}{dl} = 0')
    eqs_c.append(f'lapc1 - lap(c1,c1{dl}) = 0')
    eqs_c.append(f'lapc2 - lap(c2,c2{dl}) = 0')
    eqs_c.append(f'm1 - m1_lhs = m1_rhs')
    eqs_c.append(f'm2 - m2_lhs = m2_rhs')
    eqs_c.append(f'dt(c1) - lap(m1,m1{dl}) = - flow*adv(c1,c1{dl})')
    eqs_c.append(f'dt(c2) - lap(m2,m2{dl}) = - flow*adv(c2,c2{dl})')
    for side in ['left','right']:
        for i in '12':
            eqs_c.append(f'{side}(c{i}) = 0')
            eqs_c.append(f'{side}(m{i}{dl}) = {side}(c{i}*u{dl})')

    # u substitutions
    subs_u.append(('ρ', '+'.join([f'ρ{i}*c{i}' for i in ns])))
    # subs_u.append(('hlap(f)', '+'.join([f'd{xk}(d{xk}(f))' for xk in dims[:-1]])))
    # subs_u.append((f'lap(u,u{dl})', f'hlap(u) + d{dl}(u{dl})'))
    # for xk in dims: subs_u.append((f'surf{xk}', '+'.join([f'σ{i}{j}*lapc{i}*d{xk}(c{j})' for i in ns for j in ns if j != i])))
    for xk in dims: subs_u.append((f'surf{xk}', '+'.join([f'μ{i}_op*d{xk}(c{i})' for i in ns])))
    for xk in dims: subs_u.append((f'buoyancy{xk}', '-Bo*ρ' if xk==params.get('gravity') else '0'))
    subs_u.append(('divu', '+'.join([f'd{xk}(u{xk})' for xk in dims[:-1]]+[f'u{dl}{dl}'])))

    # u equations
    for xk in dims: 
        eqs_u.append(f'd{dl}(u{xk}) - u{xk}{dl} = 0')
        eqs_u.append(f'd{xk}(p) - Ca*lap(u{xk},u{xk}{dl}) = (1/ε)*(3/2**(1/2))*surf{xk} + buoyancy{xk}')
    eqs_u.append('divu = 0')
    nonzero_modes = ' or '.join([f'(n{xk} != 0)' for xk in dims[:-1]])
    zero_mode = ' and '.join([f'(n{xk} == 0)' for xk in dims[:-1]])
    for xk in dims[:-1]:
        for side in ['left','right']:
            eqs_u.append(f'{side}(u{xk}) = 0')
    eqs_u.append(f' left(u{dl}) = 0')
    eqs_u.append((f'right(u{dl}) = 0',nonzero_modes))
    eqs_u.append(('right(p) = 0',zero_mode))

    # def update_fields_c(solver_c): # update auxiliary fields from c1, c2
    #     for i in [1,2]:
    #         solver_c.state[f'c{i}{dl}']['g'] = solver_c.state[f'c{i}'].differentiate(dl)['g']
    #         solver_c.state[f'lapc{i}']['g'] = solver_c.evaluator.vars[f'lapc{i}_op']['g']
    #         solver_c.state[f'm{i}']['g'] = solver_c.evaluator.vars[f'm{i}_op']['g']
    #         solver_c.state[f'm{i}{dl}']['g'] = solver_c.state[f'm{i}'].differentiate(dl)['g']

    # define c problem ivp solver
    problem = de.IVP(domain, variables=c_vars+u_vars)
    for pname in params_c+params_u: problem.parameters[pname] = params[pname]
    # for f in stores_c:     problem_c.parameters[f] = stores[f]
    for sub in subs_c+subs_u:     problem.substitutions[sub[0]] = sub[1]
    for eq in eqs_c+eqs_u:
        if isinstance(eq,str): problem.add_equation(eq)
        elif isinstance(eq,tuple): problem.add_equation(eq[0],condition=eq[1])
    solver = problem.build_solver(getattr(de.timesteppers,params["timestepper"]))
    logger.info('built solver')
            
    # # define c problem ivp solver
    # problem_c = de.IVP(domain, variables=c_vars)
    # for pname in params_c: problem_c.parameters[pname] = params[pname]
    # for f in stores_c:     problem_c.parameters[f] = stores[f]
    # for sub in subs_c:     problem_c.substitutions[sub[0]] = sub[1]
    # for eq in eqs_c:       problem_c.add_equation(eq)
    # solver_c = problem_c.build_solver(getattr(de.timesteppers,params["timestepper"]))
    # logger.info('built c solver')

    # # # Define velocity stokes solver
    # problem_u = de.LBVP(domain, variables=u_vars)
    # for pname in params_u: problem_u.parameters[pname] = params[pname]
    # for f in stores_u:     problem_u.parameters[f] = stores[f]
    # for sub in subs_u:     problem_u.substitutions[sub[0]] = sub[1]
    # for eq in eqs_u:
    #     if isinstance(eq,str): problem_u.add_equation(eq)
    #     elif isinstance(eq,tuple): problem_u.add_equation(eq[0],condition=eq[1])
    # solver_u = problem_u.build_solver()
    # logger.info('built u solver')

    # initialise the fields
    uns = [f'u{xk}' for xk in dims]
    for f in solver.problem.variables: solver.state[f].set_scales(domain.dealias)
    # for f in solver_u.problem.variables: solver_u.state[f].set_scales(domain.dealias)
    # for f in stores: stores[f].set_scales(domain.dealias)
    if params.get('restart_file',None):
        # calculate shape 
        restart_series = params['restart_file'].split('/')[1]
        restart_index = int(params['restart_file'].split('/')[-1].split('-')[-1][:3])
        restart_params = pd.read_csv(f'parameters/parameters-{restart_series}.csv').iloc[restart_index]
        t_end = flt.load_data(params['restart_file'],'sim_time',group='scales',sel=np.s_[-1])[0]
        solver.sim_time = t_end        
        scaling = tuple(restart_params[f'n{d}']/params[f'n{d}'] for d in params['dims'])
        grid_slice = domain.dist.grid_layout.slices(scaling)
        for f in ['c1','c2','p']+uns:
            solver.state[f].set_scales(scaling)
            solver.state[f]['g'], = flt.load_data(params['restart_file'],f,group='tasks',sel=(-1,)+grid_slice)
            solver.state[f].set_scales(domain.dealias)
        # the remaining variables are implicitly calculated each time step, using only previous data about the concentrations, which we have
    else:
        initial_condition(domain, params,(solver.state['c1'], solver.state['c2']),ncoeffs=params['nx']//2)
    # update_fields_c(solver)
    # for f in solver.problem.variables: stores[f]['g'] = solver.state[f]['g']
    # if flow: solver_u.solve()
    # for f in solver_u.problem.variables: stores[f]['g'] = solver_u.state[f]['g']
    C10, C20 = [solver.evaluator.vars[f]['g'].min() for f in ['C1','C2']]
    logger.info('prepared fields')
    # c_vars = [f'{f}{i}{g}' for f in 'cm' for i in '12' for g in ['',dl]] + ['lapc1','lapc2']
    # u_vars = ['p'] + [f'u{xk}' for xk in dims] + [f'u{xk}{dl}' for xk in dims]
    
    # prepare saver
    if (rank == 0) and (not os.path.isdir(params["save_dir"])): flt.makedir(params["save_dir"])
    comm.Barrier()
    save_args = {}
    if params.get('save_step'): save_args['sim_dt'] = params['save_step']
    elif params.get('save_freq'): save_args['iter'] = params['save_freq']
    if params.get('max_writes'): save_args['max_writes'] = params['max_writes']
    analysis = solver.evaluator.add_file_handler(f'{params["save_dir"]}/analysis-{params["sim_name"]}',**save_args)
    for f in ['c1','c2','p','C1','C2','C3','E']+uns: analysis.add_task(f)
    if len(dims) == 3:
        save_args_slice = {params['slice_save_arg']:params['slice_save_arg_val']}
        slices = solver.evaluator.add_file_handler(f'{params["save_dir"]}/slices-{params["sim_name"]}',**save_args_slice)
        for xk in dims:
            for f in ['c1','c2','p']+uns:
                slices.add_task(f'interp({f},{xk}=0)', name=f'{f} {xk}=0')
                slices.add_task(f'integ({f},"{xk}")', name=f'{f} int_{xk}')
        for f in ['C1','C2','C3','E']: slices.add_task(f)
    logger.info('prepared savers')
    comm.Barrier()
    solver.stop_wall_time = params['stop_wall_time']
    solver.stop_iteration = params['stop_iteration']
    solver.stop_sim_time = params['stop_sim_time']

    # Flow properties for cfl and printing
    calculator = flow_tools.GlobalFlowProperty(solver, cadence=params['cfl_freq'])
    for f in ['c1','c2']:
        calculator.add_property(f,name=f'{f}mag')
        calculator.add_property(f"dt{f}", name=f'dt{f}abs')
    for f in uns:
        calculator.add_property(f'abs({f})', name=f'{f}mag')

    logger.info('prestart')
    dt = params["dt_min"]
    solver.step(dt)
    # step_solvers(dt,[solver_c,solver_u],stores,flow=flow)
    start_time = time.time()
    logger.info('started!')
    
    def cfl_check(calculator, stores, params, dt, t):
        dtmaxs = {f: 1/calculator.max(f'dt{f}abs') for f in ['c1','c2']}
        for d in params["dims"]*flow: 
            dtmaxs[f'u{d}'] = (2*params["Lx"]/params["nx"])/calculator.max(f'u{d}mag')
        dtmin = min([dtmaxs[f] for f in dtmaxs])
        if params['adaptive']:
            dt_new = dtmin/params['safety']
            dt_new = min([dt_new,dt*(1+params['cfl_frac']),1e-5 if t < params['dt_max_switch'] else params['dt_max']])
            dt = max([dt_new,params['dt_min']])
        else: dt = params['dt_min']
        return dt, dtmaxs

    def log_status(calculator, dtmaxs):
        mins = {f: calculator.min(f'{f}mag') for f in ['c1','c2']+uns}
        maxs = {f: calculator.max(f'{f}mag') for f in ['c1','c2']+uns}
        log = ' '.join(
            [f'it {solver.iteration:0>5d}',
             f'dt {dt:.1e}',
             f't {solver.sim_time:.1e}', 
             f'wall min {(time.time() - start_time)/60:.2f}',
             'mins ' + ', '.join([f'{f} {mins[f]:.3e}' for f in mins]),
             'maxs ' + ', '.join([f'{f} {maxs[f]:.3e}' for f in maxs]),
             'dtmaxs ' + ', '.join([f'dt{f} {dtmaxs[f]:.2e}' for f in dtmaxs]),
             'dtints ' + ', '.join([f'{f} {solver.evaluator.vars[f]["g"].max()-f0:.3e}' for f, f0 in zip(['C1','C2','E'],[C10,C20,0])])])
        return log

    def nan_check(comm,arr):
        broken = np.any(np.isnan(arr))
        if comm.size == 1: 
            return broken
        else:
            broken = comm.gather(broken, root=0)
            if comm.rank == 0: 
                broken = any(broken)
            broken = comm.bcast(broken,root=0)
            return broken

    # simulation loop
    while solver.ok:
        if solver.iteration % params['cfl_freq'] == 0: # test cfl
            dt, dtmaxs = cfl_check(calculator, stores, params, dt, solver.sim_time)
        if solver.iteration % params['print_freq'] == 0: # check if broken, print
            if nan_check(comm,solver.state['c1']['g']):
                logger.info('NaN!/Breaking')
                break
            log = log_status(calculator, dtmaxs)
            if params.get("test_dir"): log += ' errs ' + ', '.join([f'{f} {errs[f]:.2e}' for f in errs])
            logger.info(log)
        solver.step(dt)#step_solvers(dt,[solver_c,solver_u],stores,flow=flow)
    if not nan_check(comm,solver.state['c1']['g']):
        solver.step(dt)#step_solvers(dt,[solver,solver_u],stores,flow=flow)
    
    post.merge_analysis(f'{params["save_dir"]}/analysis-{params["sim_name"]}',cleanup=True)

