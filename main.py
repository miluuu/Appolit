import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import pdb

import modelling as md
import outputfcts
import specifics
import API

np.random.seed(1)

def main(argv, preprocessor_type, VF_approx_architecture = specifics.Gaussian_Process_Regression_GPy, optimizer_choice = 0, normalize_input = 0):
    '''Sets up model and algorithm.

        Possible inputs from terminal, in the following order:
        # dataset
        # max_iter (number of approximate policy iteration steps)
        # episode_max (samplesize in algorithm (number of samples for approximation architecture))
        # t_max
        # number_samplepaths (samplesize of data/number of samplepaths (for stochastic datasets))
        # stoch_sample_no (nr of specific samplepath in stochastic datasets)
        '''
    dataset = str(sys.argv[1])
    is_deterministic = 1 if dataset[0] == 'D' else 0
    dataset_type = 'Deterministic' if is_deterministic else 'Stochastic'
    dataset_number = int(dataset[1])  if len(sys.argv[1]) == 2 else int(dataset[1] + dataset[2])

    legend = {2: 'i', 3: 'm', 4: 't', 5: 'p', 6: 's'}
    props = {2: 10, 3: 3, 4: 2000 if is_deterministic else 100, 5: 1 if is_deterministic else 256, 6: 0} #default values (for orders of variables see below)

    #Read in values (if not given, set default values)
    for key in legend:
        if (len(sys.argv) > key and str(sys.argv[key])[0] == legend[key]):
            prop_str = ''
            for i in range(1, len(sys.argv[key])):
                prop_str += sys.argv[key][i]
            props[key] = int(prop_str)

    max_iter = props[2]
    episode_max = props[3]
    t_max = props[4]
    number_samplepaths = props[5]
    stoch_sample_no = props[6]

    #Parameters for Model
    charge_efficiency = 0.9 if is_deterministic else 1.0
    discharge_efficiency = 0.9 if is_deterministic else 1.0
    max_charge = 0.1 if is_deterministic else 5.0
    max_discharge = 0.1 if is_deterministic else 5.0
    holding_cost = 0.001

    #Sampling bounds
    price_bounds = {1:(20, 40), 2: (20, 40), 3: (20, 40), 4: (20, 40), 5: (30, 30), 6: (30, 30), 7: (30, 30), 8: (30, 30), 9: (4, 117), 10: (8, 223)} #TODO: einheitliche bounds fuer alle deterministischen Faelle?
    energy_bounds = {1:(0.04, 0.06), 2: (0, 0.07), 3: (0, 0.04), 4: (0, 0.04), 5: (0.05, 0.05), 6: (0, 0.07), 7: (0, 0.07), 8: (0, 0.04), 9: (0, 0.08), 10: (0, 0.2)}

    R_max = 100 if is_deterministic else 30
    R_start = 0 if is_deterministic else 25
    R_stepsize = 1 if (dataset_number > 4 or is_deterministic) else 0.5
    P_min = 30 if (not is_deterministic) else price_bounds[dataset_number][0]
    P_max = 70 if (not is_deterministic) else price_bounds[dataset_number][1]
    P_stepsize = 1
    E_min = 1 if (not is_deterministic) else energy_bounds[dataset_number][0]
    E_max = 7 if (not is_deterministic) else energy_bounds[dataset_number][1]
    E_stepsize = 1 if dataset_number > 4 else 0.5

    #Initialization of model
    instance = md.Model(charge_efficiency, discharge_efficiency, max_charge, max_discharge, holding_cost, is_deterministic)
    instance.set_discretization(t_max, number_samplepaths, R_start, R_max, R_stepsize, P_min, P_max, P_stepsize, E_min, E_max, E_stepsize)

    #Set exogenous information (depends in stochastic case on dataset!)
    if is_deterministic:
        energy_process = outputfcts.turn_exogenous_info_from_given_data_into_function(instance, dataset_type + ' datasets/' + dataset + '/txt/e.txt', 1, is_deterministic, stoch_sample_no)
        price_process = outputfcts.turn_exogenous_info_from_given_data_into_function(instance, dataset_type + ' datasets/' + dataset + '/txt/p.txt', 1, is_deterministic, stoch_sample_no)
        demand_process = outputfcts.turn_exogenous_info_from_given_data_into_function(instance, dataset_type + ' datasets/' + dataset + '/txt/D.txt', 0, is_deterministic, stoch_sample_no)
    else:
        #parameter choices for shocks depending on dataset number
        sig_p_dict = {1 : 25, 2 : 25, 3 : 25, 4 : 25, 5 : 0.5, 6 : 1.0, 7 : 2.5, 8 : 5.0, 9 : 5.0, 10 : 5.0, 11 : 5.0, 12 : 5.0, 13 : 1.0, 14 : 1.0, 15 : 1.0, 16 : 1.0, 17 : 1.0} #TODO: should it be 0.25 for S1-S4?
        sig_e_dict = {1 : 0, 2 : 0.5, 3 : 1.0, 4 : 1.5, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0.5, 10 : 1.0, 11 : 1.5, 12 : 2.0, 13 : 0.5, 14 : 1.0, 15 : 1.5, 16 : 0.5, 17 : 1.0}

        if dataset_number in {1,5,6,7,8}:
            energy_eps_dstr = specifics.Uniform_Distribution(instance.E_stepsize)
        else:
            sig_e = sig_e_dict[dataset_number]
            energy_eps_dstr = specifics.Pseudonormal_Distribution(0, sig_e, -3, 3, instance.E_stepsize)

        energy_process = specifics.MC_Energy(instance, energy_eps_dstr)
        demand_process = specifics.Demand(instance)
        sig_p = sig_p_dict[dataset_number]
        price_eps_dstr = specifics.Pseudonormal_Distribution(0, sig_p, -8, 8, 1)
        price_jump_dstr = specifics.Pseudonormal_Distribution(0, 50, -40, 40, 1)
        price_process = specifics.MC_Jumps_Price(instance, price_eps_dstr, 1, price_jump_dstr)

    instance.set_exogenous_information(energy_process, price_process, demand_process)
    algo = API.Algorithm(instance, max_iter, episode_max, preprocessor_type, VF_approx_architecture, optimizer_choice, normalize_input)

    algo.approximate_policy_iteration(instance, dataset, dataset_type)


if __name__ == "__main__":
    optimizer_choice = 0 # 0 scipy, 1 gridsearch, else knitro
    normalize_input = 1
    VF_approx_architecture = specifics.Gaussian_Process_Regression_GPy
    preprocessor_type =  StandardScaler if not normalize_input else Normalizer

    main(sys.argv, preprocessor_type, VF_approx_architecture, optimizer_choice, normalize_input)
