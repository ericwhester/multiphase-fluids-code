import dedalus.public as de
import numpy as np
from dedalus.tools import post
import file_tools as flt
import numpy as np
import pandas as pd
import dedalus.public as de
import logging
import sys
import os
root = logging.root
for h in root.handlers: h.setLevel("INFO") 
logger = logging.getLogger(__name__)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

series = sys.argv[1]
index = int(sys.argv[2])
save_dir = f'data/{series}'

σ12,σ13,σ23 = 0.0635531203409303, 0.498320784419641, 0.438126095239428

def create_dataframe(param_dic):
    """Convert dictionary of experiment parameters into multiindex of params for each experiment.
    Parameters paired in a tuple will be paired in the multiindex.
    E.g. {'A':[1,2], ('B','C'):([3,4],[-3,-4]),'D':[0]} ->
    A   B   C   D
    1   3  -3   0
    1   4  -4   0
    2   3  -3   0
    2   4  -4   0
    """
    tuples = []
    param_lists = {}
    for key in param_dic:
        if isinstance(key, str):
            param_lists[key] = param_dic[key]
        elif isinstance(key, tuple):
            tuples.append(key)
            param_lists[key[0]] = list(range(len(param_dic[key][0])))
            for keyi in key[1:]:
                param_lists[keyi] = [pd.NA]

    params = pd.MultiIndex.from_product(param_lists.values(), names=param_lists.keys())
    params = pd.DataFrame(index=params).reset_index()

    for tup in tuples:
        for column in tup[1:]:
            params[column] = params[tup[0]]
        for ind, column in enumerate(tup):
            params[column] = params[column].apply(lambda j: param_dic[tup][ind][j])

    return params

import glob

param_list = {
    'T' : [1e-2],
    'L' : [1e-4],
    'ν0' : [1e-6],
    'ρ0' : [1e3],
    'dρ0' : [1e2],
    'σ0' : [1e-2],
    'gr' : [10],
    'flow': [True],
    'ε' : [1e-2],
    'gravity': ['z'], # 0,'x','z': zero, horizontal, vertical pressure gradient
    'σ12': [σ12],
    'σ13': [σ13],
    'σ23': [σ23],
    'ρ1': [.5],
    'ρ2': [-.1],
    'ρ3': [6.],
    'c10':[.3],
    'Δc':[5e-2],
    'timestepper':['SBDF2'],
    'dt_min':[1e-4],
    'dt_max':[1e-4],
    'adaptive':[True],
    'cfl_freq':[50],
    'cfl_frac':[.1],
    'dt_max_switch':[.0],
    'safety':[1e1],
    'stop_sim_time':[10],
    'stop_wall_time':[24*60*60],
    'stop_iteration':[1*10**5],
    # 'restart_file':[''],
    # 'save_freq':[1000],
    'save_step':[.5],
    'print_freq':[100],
    'max_writes':[10],
    'slice_save_arg':['sim_dt'],
    'slice_save_arg_val':[.1],
    # 'pmesh':[(12,24)],
    'dims': ['xy'], # 'xyz' for 3D
    'nx':[96],
    'ny':[96],
    'nz':[192],
    'Lx':[1.],
    'dealias':[1.5],
    'save_dir': [save_dir],
    'script':[0]
}

params = create_dataframe(param_list)
L, T, ε, σ0, dρ0, ρ0, ν0, gr = [params[_] for _ in ['L','T','ε','σ0','dρ0','ρ0','ν0','gr']]
U = L/T
params['We'] = ρ0*U**2*L/σ0
params['Ca'] = ρ0*ν0*U/σ0
params['Bo'] = dρ0*gr*L**2/σ0
params['sim_name'] = ['-'.join([series,f'{i:0>3d}']) for i in params.index]
# series_restart = 'ch-3D-comparison-1'
# params['restart_file'] = [last_save_file(f'{series_restart}-{i:0>3d}') for i in range(len(params))]

params.to_csv(f'parameters/parameters-{series}.csv')

import chsb

scripts = {0:chsb}

scripts[params.loc[index]['script']].run_chs(params.loc[index])

import plot_chs
plot_chs.plot_chs(save_dir,series,params['sim_name'][index],params.loc[index])
