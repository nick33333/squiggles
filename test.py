import sys
sys.path.insert(1, 'MSMC_clustering/')
from MSMC_clustering import Msmc_clustering
from MSMC_plotting import *
import pandas as pd
import numpy as np
import pickle
import json
import os

def to_int(to_int_list, my_dict):
    for thing in to_int_list:
        my_dict[thing] = int(my_dict[thing])
    return my_dict
   
def to_float(to_float_list, my_dict):
    for thing in to_float_list:
        my_dict[thing] = float(my_dict[thing])
    return my_dict
        
def to_bool(to_bool_list, my_dict):
    for thing in to_bool_list:
        my_dict[thing] = my_dict[thing] == 'True'
    return my_dict

def to_list(to_list_list, my_dict):
    for thing in to_list_list:
        umm = thing.split(',')
        if len(umm) > 1:
            my_dict[thing] = [float(t) for t in umm]
            print(my_dict[thing])
    return my_dict

to_int_list = ['interpolation_pts',
               'manual_cluster_count',
               'omit_back_prior',
               'omit_front_prior']
to_float_list = ['mu']
to_bool_list = ['use_interpolation',
                'use_real_time_and_c_rate_transform',
                'use_time_log10_scaling',
                'use_value_normalization']
to_list_list = ['time_window']


non_user_settings =  {'algo': 'kmeans',
                      'interpolation_kind': 'linear',
                      'interpolation_pts': '100',
                      'manual_cluster_count': '7',
                      'mu': '1.4E-9',
                      'omit_back_prior': '5',
                      'omit_front_prior': '5',
                      'time_field': 'left_time_boundary',
                      'time_window': False,
                      'use_interpolation': 'True',
                      'use_real_time_and_c_rate_transform': 'True',
                      'use_time_log10_scaling': 'True',
                      'use_value_normalization': 'True',
                      'value_field': 'lambda',
                      'directory': 'data/msmc_curve_data_birds/',
                      'generation_time_path': 'data/generation_lengths/',
                      'exclude_subdirs': [],
                      'use_plotting_on_log10_scale': False,
                      'data_file_descriptor':'.txt',
                      'sep': '\t'}
user_settings =  {'algo': 'kmeans', 'interpolation_kind': 'linear', 'interpolation_pts': '100', 'manual_cluster_count': '7', 'mu': '1.4E-9', 'omit_back_prior': '5', 'omit_front_prior': '5', 'time_field': 'time', 'time_window': False, 'use_interpolation': 'True', 'use_real_time_and_c_rate_transform': 'True', 'use_time_log10_scaling': 'True', 'use_value_normalization': 'True', 'value_field': 'NE', 'directory': 'static/uploads/', 'generation_time_path': 'data/generation_lengths/', 'exclude_subdirs': [], 'use_plotting_on_log10_scale': False, 'sep': '\t'}


non_user_settings = to_float(to_float_list=to_float_list,
                             my_dict=non_user_settings)
non_user_settings = to_int(to_int_list=to_int_list,
                             my_dict=non_user_settings)
non_user_settings = to_bool(to_bool_list=to_bool_list,
                             my_dict=non_user_settings)
non_user_settings = to_list(to_list_list=to_list_list,
                             my_dict=non_user_settings)

non_user_settings = to_float(to_float_list=to_float_list,
                             my_dict=non_user_settings)
non_user_settings = to_int(to_int_list=to_int_list,
                             my_dict=non_user_settings)
non_user_settings = to_bool(to_bool_list=to_bool_list,
                             my_dict=non_user_settings)
non_user_settings = to_list(to_list_list=to_list_list,
                             my_dict=non_user_settings)
print("non_user_settings: ", non_user_settings)
m_obj_base = Msmc_clustering(**non_user_settings)
# m_obj_user = Msmc_clustering(**user_settings)
