import numpy as np

class State:
    '''Element of the state space'''
    def __init__(self, storage, energy, price, demand):
        self.storage = storage
        self.energy = energy
        self.price = price
        self.demand = demand

    def convert_to_array(self):
        return np.array([self.storage, self.energy, self.price, self.demand])

class Decision:
    '''Decision vector, in notation from paper: x^wd, x^gd, x^rd, x^wr, x^gr, x^rg) (all non-negative)'''
    def __init__(self, dec_wd, dec_rd, dec_gd, dec_wr, dec_gr, dec_rg):
        self.dec_wd = dec_wd
        self.dec_gd = dec_gd
        self.dec_rd = dec_rd
        self.dec_wr = dec_wr
        self.dec_gr = dec_gr
        self.dec_rg = dec_rg

    #NOTE: See Readme.pdf: In benchmark datasets the order of the decision x_gd and x_rd is swapped! (compared to "Does Anything Work")
    def convert_to_array (self):
        return np.array([self.dec_wd, self.dec_rd, self.dec_gd, self.dec_wr,self.dec_gr,self.dec_rg])

def convert_array_to_decision(array):
    '''Takes array, returns corresponding Decision'''
    return Decision(array[0], array[1], array[2], array[3], array[4], array[5])


class Model:
    '''Model of the storage facility, and of the exogenous process'''

    def __init__(self, charge_efficiency, discharge_efficiency, max_charge, max_discharge, holding_cost, is_deterministic):
        '''Set attributes related to storage facility'''
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.holding_cost = holding_cost
        self.is_deterministic = is_deterministic
        self.rho = discharge_efficiency if is_deterministic else 0.98

    def set_discretization(self, t_max, number_samplepaths, R_start, R_max, R_stepsize, P_min, P_max, P_stepsize, E_min, E_max, E_stepsize):
        '''Set attributes related to the discretization'''
        self.t_max = t_max
        self.number_samplepaths = number_samplepaths
        self.R_start = R_start
        self.R_max = R_max # R_min = 0
        self.R_stepsize = R_stepsize
        self.P_min = P_min
        self.P_max = P_max
        self.P_stepsize = P_stepsize
        self.E_min = E_min
        self.E_max = E_max
        self.E_stepsize = E_stepsize

    def set_exogenous_information(self, energy_process, price_process, demand_process):
        '''Set model of exogenous information, i.e. of energy(wind), price and demand

        the variables are given as classes, each with sample functions which take time t and the respective previous value as parameters (see specifics.py)
        '''
        self.energy_process = energy_process if self.is_deterministic else energy_process.sample
        self.price_process = price_process if self.is_deterministic else price_process.sample
        self.demand_process = demand_process if self.is_deterministic else demand_process.sample

    def get_state(self, state, t):
        '''Get exogenous information (energy, price, demand) at time t (if stochastic: sample), update directly in state. Demand is considered to be deterministic.'''
        state.energy = self.energy_process(t, state.energy)
        state.price = self.price_process(t, state.price)
        state.demand = self.demand_process(t)

    def transition(self, algo, decision, state):
        '''Transition dynamics

        Calculates effect of decision vector on storage, no dependence on exogenous information
        Returns new storage
        '''
        new_storage = state.storage + self.charge_efficiency * (decision.dec_wr + decision.dec_gr) - (decision.dec_rd + decision.dec_rg)
        if new_storage < 0 and abs(new_storage) < algo.feas_eps:
            return 0
        if new_storage > self.R_max and abs(self.R_max - new_storage) < algo.feas_eps:
            return self.R_max
        return new_storage

    def contribution(self, algo, decision, state):
        '''Returns realized profit'''
        return state.price * (state.demand + self.rho * decision.dec_rg - decision.dec_gr - decision.dec_gd) - self.holding_cost * self.transition(algo, decision, state)

    def is_feasible(self, algo, decision, state):
        '''Determines if element lies in feasible action space (for given state). If not, returns the number of the first violated constraint and the discrepancy of the violation.'''
        if decision.dec_rd + decision.dec_rg > min(state.storage, self.max_discharge) + algo.feas_eps:
            return (2 , decision.dec_rd + decision.dec_rg - min(state.storage, self.max_discharge))
        if decision.dec_wr + decision.dec_gr > min(self.R_max - state.storage, self.max_charge) + algo.feas_eps:
            return (3 , decision.dec_wr + decision.dec_gr -min(self.R_max - state.storage, self.max_charge))
        if decision.dec_wr + decision.dec_wd > state.energy + algo.feas_eps:
            return (4, decision.dec_wr + decision.dec_wd - state.energy)
        if abs(decision.dec_wd + self.discharge_efficiency * decision.dec_rd + decision.dec_gd - state.demand) > algo.feas_eps:
            return (1 , state.demand - (decision.dec_wd + self.discharge_efficiency * decision.dec_rd + decision.dec_gd))

        return 0

    def initial_guess_for_policy_improvement(self, state):
        wd = min(state.demand, state.energy)
        rd = min(max(state.storage, 0), self.max_discharge, (state.demand - wd) / self.discharge_efficiency)
        gd = max(0, state.demand - wd - rd * self.discharge_efficiency)
        wr = min(self.max_charge, state.energy - wd, self.R_max - state.storage)
        rg = 0
        gr = 0

        return (wd, rd, gd, wr, gr, rg)
