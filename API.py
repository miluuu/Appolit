#Implementation of the approximate policy evaluation and policy iteration

import matplotlib.pyplot as plt
import numpy as np
import modelling as md
from scipy.optimize import minimize, brute
import policy_improvement
import policy_improvement_VF_only
import policy_improvement_contr_only
from math import floor
import outputfcts
import pdb

#TODO:DELETE
#np.seterr(invalid='raise')

class Algorithm:
    '''Performs approximate policy iteration'''
    def __init__(self, model, max_iter, episode_max, preprocessor_type, VF_approx_architecture, optimizer_choice = 0, normalize_input=0):
        self.max_iter = max_iter        #N (in Paper)
        self.episode_max = episode_max  #M (in Paper)
        self.VF_approximation = {t:  VF_approx_architecture() for t in range(model.t_max)} #updated in every iteration
        self.preprocessor = {t: preprocessor_type() for t in range(model.t_max)}
        self.optimizer_choice = optimizer_choice
        self.ftol = 1e-15 if optimizer_choice in {0,1} else 1e-15 # precision goal for the value of objective function in the stopping criterion for the optimizer
        self.eps = 1e-15 #step size for finite-difference derivative estimates
        self.feas_eps = 1e-06
        self.initial = 0 #Hilfsvariable
        self.normalize_input = normalize_input

    def scipy_policy_improvement_for_this_state(self, model, state, t, initial_guess, VF_initial_guess, contr_initial_guess, iteration = 0):
        '''Policy improvement step with scipy'''
        #Optimization of objective function, value function and contribution function with scipy.minimize
        const1 =  min(state.storage, model.max_discharge)
        print("const1, storage, max_discharge:", const1, state.storage, model.max_discharge)
        const2 = min(model.R_max - state.storage, model.max_charge)
        print("const2, R_max, storage, max_discharge:", const2, model.R_max, state.storage, model.max_discharge)
        const3 = min(state.energy, state.demand)

        def obj_fct(x):
    #        pdb.set_trace()
            decision = md.convert_array_to_decision(x)
            input = [model.transition(self, decision, state), state.energy, state.price]
            factor = np.linalg.norm(input) if self.normalize_input else 1
            return - model.contribution(self, decision, state) - factor * self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([input])))

        def VF_obj_fct(x):
            decision = md.convert_array_to_decision(x)
            input = np.array([model.transition(self, decision, state), state.energy, state.price])
            factor = np.linalg.norm(input) if self.normalize_input else 1
            return  - factor *self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([input])))

        def contr_obj_fct(x):
            return - model.contribution(self, md.convert_array_to_decision(x), state)

        bnds = ((0, const3), (0, const1), (0, state.demand), (0, const2), (0, const2), (0, const1)) #approximation: open intervals (since closed intervals not available as bounds)
        beta_d = model.discharge_efficiency
        cons = ({'type': 'eq', 'fun': lambda x: x[0] + beta_d * x[1] + x[2]  - state.demand},
                {'type': 'ineq', 'fun': lambda x: - x[1] - x[5] + const1},
                {'type': 'ineq', 'fun': lambda x: - x[3] - x[4] + const2},
                {'type': 'ineq', 'fun': lambda x: - x[0] - x[3] + state.energy})


        print("Initial guess: ", initial_guess)
        print("Initial guess is feasible:", model.is_feasible(self, md.convert_array_to_decision(initial_guess), state))

        solution = minimize(obj_fct, initial_guess, method = 'SLSQP', bounds = bnds, constraints = cons, options = {'eps': self.eps, 'ftol': self.ftol}).x
        VF_solution = minimize(VF_obj_fct, VF_initial_guess, method = 'SLSQP',bounds = bnds, constraints = cons, options = {'ftol': self.ftol}).x
        contr_solution = minimize(contr_obj_fct, contr_initial_guess, method = 'SLSQP',bounds = bnds, constraints = cons, options = {'ftol': self.ftol}).x
        return solution, VF_solution, contr_solution


    def policy_improvement_for_this_state(self, model, state, t, iteration = 0):
        '''Policy improvement step for one particular state at time t (policy stored as dictionary)'''
        const1 =  min(state.storage, model.max_discharge)
        const2 = min(model.R_max - state.storage, model.max_charge)
        const3 = min(state.energy, state.demand)
#        if t==1:
#            pdb.set_trace()

        if self.optimizer_choice == 0:
            # Set initial guess
            initial_guess = model.initial_guess_for_policy_improvement(state)
            res = self.scipy_policy_improvement_for_this_state(model, state, t, initial_guess, initial_guess, initial_guess, iteration)
            solution = res[0]
            VF_solution = res[1]
            contr_solution = res[2]
###########################
        #use gridsearch
        elif self.optimizer_choice == 1:
            #TODO: Use finish function?
            params = (state.storage, state.energy, state.price, state.demand, t)

            def infeasibility_indicator(x, *params):
                x0, x1, x2, x3, x4, x5 = x
                stor, ener, price, dem, t = params
                feas =  model.is_feasible(self, md.Decision(x0, x1, x2, x3, x4, x5), md.State(stor, ener, price, dem))
                if feas != 0:
                    if feas[0] == 1 and abs(feas[1]) < 0.01:
                        return 0.1
                    return 1
                return 0

            #NOTE: The objective functions are not the same as for the  other solvers, as they include the constraints!
            def objective(x, *params):
                x0, x1, x2, x3, x4, x5 = x
                stor, ener, price, dem, t = params
                state = md.State(stor, ener, price, dem)
                decision = md.Decision(x0, x1, x2, x3, x4, x5)
                input = [model.transition(self, decision, state), state.energy, state.price]
                factor = np.linalg.norm(input) if self.normalize_input else 1
                return (infeasibility_indicator(x, *params) * 1e6 - model.contribution(self, decision, state)  - factor*self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([input]))))

            def contr_objective(x, *params):
                x0, x1, x2, x3, x4, x5 = x
                stor, ener, price, dem, t = params
                state = md.State(stor, ener, price, dem)
                decision = md.Decision(x0, x1, x2, x3, x4, x5)
                return (infeasibility_indicator(x, *params) * 1e6 - model.contribution(self, decision, state))

            def VF_objective(x, *params):
                x0, x1, x2, x3, x4, x5 = x
                stor, ener, price, dem, t = params
                state = md.State(stor, ener, price, dem)
                decision = md.Decision(x0, x1, x2, x3, x4, x5)
                input = [model.transition(self, decision, state), state.energy, state.price]
                factor = np.linalg.norm(input) if self.normalize_input else 1
                return (infeasibility_indicator(x, *params) * 1e6  - factor*self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([input]))))


            #TODO: different stepsize?
            stepsize =  0.1 * max(const1, const2, const3)
            print("const1:", const1, ", const2:", const2, ", const3:", const3)
            rranges = (slice(0, const3 + stepsize, stepsize) , slice(0, const1 + stepsize, stepsize), slice(0, state.demand + stepsize, stepsize), slice(0, const2 + stepsize, stepsize), slice(0, const2 + stepsize, stepsize), slice(0, const1 + stepsize, stepsize))
        #    pt = outputfcts.progress_timer(description = 'Progress', n_iter = 1)
            solution = brute(objective, rranges, args=params, finish = None)
            contr_solution = brute(VF_objective, rranges, args=params, finish = None)
            VF_solution = brute(contr_objective, rranges, args=params, finish = None)

            res = self.scipy_policy_improvement_for_this_state(model, state, t, solution, VF_solution, contr_solution, iteration)
            solution = res[0]
            VF_solution = res[1]
            contr_solution = res[2]

        #    pt.update()
        #    pt.finish()
################################
        #solve maximization problem with Artelys Knitro Solver (student free trial version)
        else:
        #TODO: für normalize_input case anpassen!
            if self.normalize_input:
                print("knitro policy improvement muss noch normalize_input angepasst werden!")
                return
            solution = policy_improvement.maximize(model, self, state, lambda decision: self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([[model.transition(self, md.convert_array_to_decision(decision), state), state.energy, state.price]]))), t) #GPy version
            VF_solution = policy_improvement_VF_only.maximize(model, state, lambda decision: self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([[model.transition(self, md.convert_array_to_decision(decision), state), state.energy, state.price]]))), t) #GPy version
            contr_solution = policy_improvement_contr_only.maximize(model, self, state, t) #GPy version

        print("Iteration", iteration, " at time", t, solution)
        print("State: st, p, e, d ", state.storage, state.price, state.energy, state.demand)
        print("Post-decision storage after", t, ":", model.transition(self, md.convert_array_to_decision(solution), state))
        print("Solution is feasible:", model.is_feasible(self, md.convert_array_to_decision(solution), state))
        input1 = [model.transition(self, md.convert_array_to_decision(solution), state), state.energy, state.price]
        norm1 = np.linalg.norm(input1) if self.normalize_input else 1
        input2 = [model.transition(self, md.convert_array_to_decision(VF_solution), state), state.energy, state.price]
        norm2 = np.linalg.norm(input2) if self.normalize_input else 1
        print("VF_approximation at solution:", norm1*self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([input1])))) #GPy version
        print("Contribution at solution:", model.contribution(self, md.convert_array_to_decision(solution), state))
        print("Optimal contribution obtained at ", contr_solution, "with contribution value:", model.contribution(self, md.convert_array_to_decision(contr_solution), state))
        print("Optimal VF obtained at ", VF_solution, "with VF value:", norm2*self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([input2]))))
        return md.convert_array_to_decision(solution)

    def approx_policy_evaluation_and_update_VF(self, model, iteration):
        '''Performs approximate policy evaluation step, returns a dictionary with functions (values) for every time t(keys) (an approximation of the post-decision value function for each time t).'''
        agg_values = {m: np.zeros(model.t_max) for m in range(self.episode_max)} #for all m and for all t, stores v^m_t (in Paper), aggregated undiscounted sum of contributions from point t on in sample m
        pdstates = {m: {t: np.array(3) for t in range(model.t_max)} for m in range(self.episode_max)} # store all post decision states as arrays (do not include demand, as demand is modelled deterministically here)

        #Simulated samplepaths
        for m in range(self.episode_max):
            #Sample initial state.
            if model.is_deterministic:
#                state_m = md.State(np.random.uniform(0, model.R_max), 0,0,0)# TODO: maybe change back?
                state_m = md.State(0, 0, 0, 0)
            else:
                state_m = md.State(np.random.choice(np.arange(0, model.R_max, model.R_stepsize)), np.random.choice(np.arange(model.P_min, model.P_max, model.P_stepsize)), np.random.choice(np.arange(model.E_min, model.E_max, model.E_stepsize)),0)
            #Go through sample paths.
            for t in range(model.t_max):
                model.get_state(state_m, t) #update exogenous information (excluding storage value)
                if model.is_deterministic: #TODO: change?
                    state_m.energy = np.random.uniform(model.E_min, model.E_max)
                    state_m.price = np.random.uniform(model.P_min, model.P_max)
                if iteration!= 0 or self.initial==1:
                    print("Episode ", m)
                    decision_m =  self.policy_improvement_for_this_state(model, state_m, t, iteration)
                else:
                    # Set initial policy (randomized, while satisfying feasibility constraints)
                    wd = np.random.uniform(0, min(state_m.demand, state_m.energy))
                    rd = np.random.uniform(0, min(state_m.storage, model.max_discharge, (state_m.demand - wd))/ model.discharge_efficiency)
                    gd = max(0, state_m.demand - wd - rd * model.discharge_efficiency) #not randomized, in order to satisfy demand/be feasible
                    wr = np.random.uniform(0, min(state_m.energy - wd, model.max_charge, model.R_max - state_m.storage))
                    rg = np.random.uniform(0, max(0, min(state_m.storage - rd, model.max_discharge - rd)))
                    gr = np.random.uniform(0, max(0, min((model.R_max - state_m.storage - model.charge_efficiency * wr + rd + rg)/ model.charge_efficiency, model.max_charge - wr)))

                    decision_m = md.Decision(wd, rd, gd, wr, gr, rg)

                current_contribution = model.contribution(self, decision_m, state_m)
                state_m.storage = model.transition(self, decision_m, state_m) #update storage value
                pdstates[m][t] = np.array([state_m.storage, state_m.energy, state_m.price])

                for s in range(t + 1):
                    agg_values[m][s] += current_contribution

        #Calculate the VF Approximations:
        for t in range(model.t_max):
#            pdb.set_trace()
            pd_samples_t = np.array([pdstates[m][t] for m in range(self.episode_max)])
            value_samples_t = np.array([[agg_values[m][t]] for m in range(self.episode_max)]) # GPy version
    #        value_samples_t = np.array([agg_values[m][t] for m in range(self.episode_max)])

            if self.normalize_input:
                norms = np.sum(np.abs(pd_samples_t)**2,axis=-1)**(1./2) #calculate l2-norm for each pdstate
                value_samples_t = np.array([[value_samples_t[i][0] /norms[i]] for i in range(self.episode_max)])
            pd_samples_t_preprocessed = self.preprocessor[t].fit_transform(pd_samples_t)

            self.VF_approximation[t].fit(pd_samples_t_preprocessed, value_samples_t)
            print("for iteration", iteration, "fitted approximation for time", t)
            self.initial = 1
        #    pdb.set_trace()

    def approximate_policy_iteration(self, model, dataset, dataset_type):
        '''Performs approximate policy iteration step

        (policy improvement is invoked within policy evaluation step)
        To apply the final policy to a state, policy_improvement_for_this_state needs to be applied first.
        '''


        for iteration in range(self.max_iter):
            print("ITERATION ", iteration)
            res = self.approx_policy_evaluation_and_update_VF(model, iteration)

            # Evaluate current optimal policy.
            solution_storage = np.zeros(model.t_max)
            solution_storage = self.evaluate_policy(model, dataset_type + ' datasets/' + dataset + '/txt/e.txt', dataset_type + ' datasets/' + dataset + '/txt/p.txt', dataset_type + ' datasets/' + dataset + '/txt/D.txt')
            for k in range(model.number_samplepaths):
                outputfcts.plot_data_and_save(model, solution_storage[k], 'R' + '-' + dataset, 'Results/' + dataset + '/solution_storage plot, ' + dataset + ', ' + str(iteration) + 'out of' + str(self.max_iter) + ' iterations, m = ' + str(self.episode_max) + ', sample ' + str(k) + ' .pdf')
                outputfcts.save_data_in_file(solution_storage[k], 'Results/' + dataset + '/solution_storage, ' + dataset + ', '  + str(iteration) + ' out of' + str(self.max_iter) + ' iterations, m = ' + str(self.episode_max) + ', sample ' + str(k) + ' .txt')

            # Plot slices of value function
            plot_bool_3D = 0 #TODO: als Parameter uebergeben?
            plot_bool_2D = 0

            if plot_bool_3D == 1:

                plot_tmax = 10 if model.t_max > 11 else model.t_max #TODO: set accordingly
                for t in range(plot_tmax):
                    fig=plt.figure()
                    ax = fig.add_subplot(111, projection = '3d')
                    ax.set_xlabel('x_wr')
                    ax.set_ylabel('R')
                    ax.set_zlabel('VF')
                    X_test = np.array([[x for i in np.arange(0, model.R_max, 1)] for x in np.arange(0, model.max_discharge, 0.001)])
                    Y_test = np.array([[y for y in np.arange(0, model.R_max, 1)] for i in np.arange(0, model.max_discharge, 0.001)])
                    if self.normalize_input:
                        print("Z_test muss noch an normalize_input angepasst werden!")
                        return
                        #Z_test = np.array([[  self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([[model.transition(self, md.convert_array_to_decision([0.019, 0, 0, x, 0, 0]), md.State(y, 0.05, 23, 0.019)), 0.05, 28]])))[0][0] for y in np.arange(0, model.R_max, 1)] for x in np.arange(0, model.max_discharge, 0.001)])  #GPy Version
                    else:
                        Z_test = np.array([[  self.VF_approximation[t].predict(self.preprocessor[t].transform(np.array([[model.transition(self, md.convert_array_to_decision([0.019, 0, 0, x, 0, 0]), md.State(y, 0.05, 23, 0.019)), 0.05, 28]])))[0][0] for y in np.arange(0, model.R_max, 1)] for x in np.arange(0, model.max_discharge, 0.001)])  #GPy Version
        #            Z_test = np.array([[  self.VF_approximation[t].predict(np.array([[model.transition(self, md.convert_array_to_decision([0.019, 0, 0, x, 0, 0]), md.State(y, 0.05, 28, 0.019)), 0.05, 28]]))[0] for y in np.arange(0, model.R_max, 1)] for x in np.arange(0, model.max_discharge, 0.001)])
                    ax.plot_surface(X_test,Y_test,Z_test, rstride = 10, cstride = 10)
                    fig.canvas.set_window_title('VF approximation at time ' + str(t))
                    fig.savefig("Results/" + dataset + "/VF_plot, " + str(self.max_iter) + "iterations, " + str(self.episode_max) + "samples, time" + str(t) +  " at iteration " + str(iteration) + ".pdf", bbox_inches='tight')
                    plt.show()

            if plot_bool_2D == 1:
                for t in range(model.t_max):
                    plt.plot([model.transition(self, md.Decision(1, 0, 0, 3, 0, 1), md.State(R, 6, 37, 2)) for R in range(0, model.R_max, 1)], [self.VF_approximation[t].predict(self.preprocessor[t].transform([[model.transition(self, md.Decision(1, 0, 0, 3, 0, 1), md.State(R, 6, 37, 2)), 6, 37]]))[0] for R in range(0, model.R_max, 1)])
                    plt.xlabel('Storage R')
                    plt.ylabel('Post-decision VF')
                    plt.show()


    def evaluate_policy(self, model, energy_file, price_file, demand_file):
        '''Evaluates policy based on given sample paths from Powell datasets and returns the average total contribution.'''
        energy_samples = np.loadtxt(energy_file)
        price_samples = np.loadtxt(price_file)
        demand = np.loadtxt(demand_file)
        storage_values = {k: np.zeros(model.t_max) for k in range(model.number_samplepaths)}
        contributions = np.zeros(model.number_samplepaths)

        if model.number_samplepaths == 1:
            storage_values[0], contributions[0] = self.evaluate_policy_for_one_iteration(model, energy_samples, price_samples, demand, 0)
        else:
            for k in range(model.number_samplepaths):
                storage_values[k], contributions[k] = self.evaluate_policy_for_one_iteration(model, energy_samples[:model.t_max, k], price_samples[:model.t_max, k], demand[:model.t_max], k)

        av_contribution = np.sum(contributions) / model.number_samplepaths
        print("The average total contribution (over %d sample(s)) is: %f" %(model.number_samplepaths, av_contribution))
        return storage_values

    def evaluate_policy_for_one_iteration(self, model, data_energy, data_price, data_demand, sample_no):
        '''Evaluates policy for a single given sample path and returns the total contribution.'''
        state = md.State(model.R_start,0,0,0)
        contribution_sum = 0
        storage_values = np.zeros(model.t_max)

        for t in range(model.t_max):
            state.energy = data_energy[t]
            state.price = data_price[t]
            state.demand = data_demand[t] if model.is_deterministic else floor(data_demand[t])
            decision = self.policy_improvement_for_this_state(model, state, t)
            contribution_sum += model.contribution(self, decision, state)
            state.storage = model.transition(self, decision, state)
            storage_values[t] = state.storage

        if model.number_samplepaths > 1:
            print("EVALUATION OF CURRENT POLICY FOR SAMPLE %d: The total contribution for sample %d is: %f " %(sample_no, sample_no, contribution_sum))
        return storage_values, contribution_sum
