import datetime
import multiprocessing as mp
import os
import time

import h5py
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from numba import cuda
from sklearn.preprocessing import MinMaxScaler

from GA_ABC import GA_Individual, GA_Model
from GA_Support import make_results_json
from qae import main


class Individual(GA_Individual):
    """
    Data structure for the ansatz.
    """
    def __init__(self, n_qubits, n_moments, gates_arr, gates_probs, rng_seed):
        """
        Initializes the inidivual object, all individuals (ansatz) are members of this class.
            - n_qubits: Number of qubits in the ansatz.
            - n_moments: Number of moments in the ansatz.
            - gates_arr: Number of possible states on a qubit. 
                Defined as 1 + n_undirected_gates + 2n_directed_gates.
                
        TODO: change params to allow for multi-param gates
        """
        self.n_qubits = n_qubits
        self.n_moments = n_moments
        self.gates_arr = gates_arr
        self.gates_probs = gates_probs
        self.rng = np.random.default_rng(seed=rng_seed)
        
        self.ansatz = []
        self.ansatz_qml = []
        self.params = []
        self.generate() # Should this be done automatically?
        
    def __len__(self):
        return len(self.ansatz[0])
    
    def __getitem__(self, key):
        if type(key) is tuple:
            return self.ansatz[int(key[0])][int(key[1])]
        else:
            return self.ansatz[int(key)]

    def __setitem__(self, key, value):
        """
        TODO: fix implementation (need to set whole moments?)
        """
        if type(key) is tuple:
            self.ansatz[int(key[0])][int(key[1])] = value
        else:
            self.ansatz[int(key)] = value
            
    def __str__(self):
        return str(self.ansatz)
        
    def generate(self):
        """
        Generates the ansatz stochastically based on the other member variables.
        """
        for j in range(self.n_moments):
            self.ansatz.append(dict.fromkeys(range(self.n_qubits), 0))
            ix = self.rng.permutation(self.n_qubits) # which qubit to pick first
            for i in ix:
                if self.ansatz[j][i] != 0:
                    continue
                k = self.rng.choice(self.gates_arr, p=self.gates_probs)
                
                # if k.find('_') < 0:
                if k[0] != 'C': # change to search for _ by modifying the gates_arr?
                    self.ansatz[j][i] = k
                    continue
                    
                q_p_arr = self.rng.permutation(self.n_qubits) # qubit_pair_array for 2-qubit gates
                for q_p in q_p_arr:
                    if self.ansatz[j][q_p] != 0 or q_p == i:
                        continue
                    
                    direction = self.rng.permutation(['_C', '_T'])
                    self.ansatz[j][i] = k + direction[0] + f'-{q_p}'
                    self.ansatz[j][q_p] = k + direction[1] + f'-{i}'
                    break
                    
                if self.ansatz[j][i] == 0:
                    self.ansatz[j][i] = self.rng.choice(self.gates_arr[:-1])
        
    def convert_to_qml(self):
        """
        Converts the ansatz into a general format for the QML.

            moment_dict example: **_C (_T) means current qubit is control (target) qubit of 2-qubit gate**
                {'I': [],
                 'RX': [],
                 'RY': [],
                 'RZ': [],
                 'CNOT': []}
        """
        # print(self.ansatz)
        self.ansatz_qml = []
        for j in range(len(self.ansatz)):
            moment_dict = {i: list() for i in self.gates_arr}
            stored_i = []
            for i in range(self.n_qubits):
                _ix = self.ansatz[j][i].find('_')
                if _ix < 0:
                    moment_dict[self.ansatz[j][i]].append(i)
                else:
                    if i in stored_i:
                        continue
                    _1ix = self.ansatz[j][i][_ix+1]
                    q_p = int(self.ansatz[j][i][-1])
                    stored_i.append(q_p)
                    if _1ix == 'C':
                        moment_dict[self.ansatz[j][i][:_ix]].append([i, int(q_p)])
                    else:
                        moment_dict[self.ansatz[j][i][:_ix]].append([int(q_p), i])

            for k in moment_dict.keys():
                if len(moment_dict[k]) == 0 or k == 'I': # add in identity gates?
                    continue
                if len(k) <= 2:
                    self.ansatz_qml.append(f"qml.broadcast(qml.{k}, wires={moment_dict[k]}, pattern='single', parameters=params[{len(self.params)}:{len(self.params)+len(moment_dict[k])}])")
                    # change to allow for one-qubit gates with 0 or 2+ params
                    for i in range(len(moment_dict[k])):
                        self.params.append(0.0)
                elif len(k) > 2: # Assumes the 2-qubit gates have no parameters, which is not generally true
                    self.ansatz_qml.append(f"qml.broadcast(qml.{k}, wires={np.array(moment_dict[k]).flatten(order='C').tolist()}, pattern={moment_dict[k]})")
                    # change to allow for two-qubit gates with 1+ params
            
            
class Model(GA_Model):
    """
    Container for all the logic of the GA Model.
    
    TODO: change the I assignment to a random assignment. check for back-to-back CNOTs bc compile to I
    """
    
    def __init__(self, config):
        """
        TODO: write out explanations for hyperparams
        """
        ### hyperparams for GA ###
        self.backend_type = config['backend_type']
        
        self.n_qubits = config['n_qubits']
        self.n_moments = config['n_moments']
        self.gates_arr = config['gates_arr']
        self.gates_probs = config['gates_probs']
        self.pop_size = config['pop_size']
        self.n_winners = config['n_winners']
        self.n_mutations = config['n_mutations']
        self.n_steps = config['n_steps']
        self.best_perf = [0, [], 0]
        
        ### hyperparams for qae ###
        self.latent_qubits = config['latent_qubits']
        self.n_shots = config['n_shots']
        self.events = config['events']
        
        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.rng_seed = config['seed']
        self.rng = np.random.default_rng(seed=self.rng_seed)
        
        self.population = []
        self.fitness_arr = []
        for _ in range(self.pop_size):
            self.population.append(Individual(self.n_qubits, self.n_moments, self.gates_arr, self.gates_probs, self.rng_seed))
            self.fitness_arr.append(0)
            print(self.population[_])
    
    def evolve(self):
        """
        Evolves the GA.
        """
        step = 0
        results = {
            'full_population': None,
            'full_fitness': None,
            'best_ansatz': None,
            'best_fitness': 0.0,
        }
        while True:
            print(f'GA iteration {step}')
            self.fitness_arr = [0 for i in self.population]
            self.evaluate_fitness(step)
            results['full_population'] = [i.ansatz for i in self.population]
            # results['full_fitness'] = self.fitness_arr
            results['best_ansatz'] = self.best_perf[1].ansatz
            results['best_fitness'] = self.best_perf[0]
            
            parents = self.select()
            for i in range(self.pop_size//self.n_winners):
                self.mate(parents)
                
            print(f'Best Fitness: {self.best_perf[0]}, Best ansatz: {self.best_perf[1]}')
            
            if step > 20 and step%self.n_steps == 0:
                if (step - self.best_perf[2]) > self.n_steps:
                    break
            make_results_json(results, self.start_time, self.script_path, step)
            step += 1
        print('filename is: ', make_results_json(results, self.start_time, self.script_path, step, final_flag=True))
    
    def evaluate_fitness(self, gen):
        """
        Evaluates the fitness level of all ansatz. Runs the QML optimization task.
        
        TODO: change to do per given ansatz (so we don't have to train every ansatz).
            -> make so fitness_arr can be shorter than population
        """
        ix = 0
        args_arr = []
        for p in self.population:
            args = []
            p.convert_to_qml()
            args.extend((p.ansatz_qml, p.ansatz, p.params, self.events, self.n_qubits, self.latent_qubits, self.rng_seed, ix, gen, self.start_time))
            args_arr.append(args)
            ix += 1
        
        start_time = time.time()
        with mp.get_context("spawn").Pool(processes=len(args_arr)) as pool:
            self.fitness_arr = pool.starmap(main, args_arr)
        end_time = time.time()
        exec_time = end_time-start_time
        print(f'QML Optimization in {exec_time:.2f} seconds')
        
        if self.best_perf[0] < np.amax(self.fitness_arr):
            self.best_perf[0] = np.amax(self.fitness_arr)
            self.best_perf[1] = self.population[np.where(self.fitness_arr == np.amax(self.fitness_arr))[0][0]]
            self.best_perf[2] = gen
    
    def select(self):
        """
        Picks the top performing ansatz from a generation to mate and mutate for the next generation.
        """
        winner_arr = []
        for i in range(self.n_winners):
            winner = self.population[np.where(self.fitness_arr == np.amax(self.fitness_arr))[0][0]]
            winner_arr.append(winner)
            self.fitness_arr.remove(np.amax(self.fitness_arr))
            
        return winner_arr
    
    def mate(self, parents):
        """
        Swaps the moments of ansatz.
        
        TODO: add in functionality to swap at the gate level
        """
        # Randomly shuffle parent array
        parents_copy = [i for i in parents]
        ix_arr = self.rng.permutation(len(parents))
        for i in range(len(parents)):
            parents[i] = parents_copy[ix_arr[i]]
        
        # Choose moments to swap for each parent
        chosen_moments = np.zeros(len(parents))
        for p in range(len(parents)):
            chosen_moments[p] = self.rng.integers(len(parents[p]))
        
        # Create an array of children for the output
        children = [i for i in parents]
        
        # Perform the swap with neighboring parents 
        # (and a 3-way swap if odd number of parents)
        for i in range(len(parents)):
            if i%2 != 1:
                continue
            children[i][chosen_moments[i]] = parents[i-1][chosen_moments[i-1]]
            children[i-1][chosen_moments[i-1]] = parents[i][chosen_moments[i]]
        if len(parents)%2 != 0:
            triswap_ix = self.rng.integers(len(parents)-1)
            children[-1][chosen_moments[-1]] = parents[triswap_ix][chosen_moments[triswap_ix]]
            children[triswap_ix][chosen_moments[triswap_ix]] = parents[-1][chosen_moments[-1]]
        
        for child in children:
            self.mutate(child)
    
    def mutate(self, ansatz):
        """
        Mutates a single ansatz by modifying n_mutations random qubit(s) at 1 random time each.
        
        TODO: add in functionality to add or remove moments from an ansatz
        """
        print(ansatz)
        double_swap_flag = 0
        for i in range(self.n_mutations):
            j = self.rng.integers(self.n_moments)
            i = self.rng.integers(self.n_qubits)
            k = self.rng.choice(self.gates_arr, p=self.gates_probs)

            if ansatz[j][i][0] == 'C':
                double_swap_flag += 1
                k_p = self.rng.choice(self.gates_arr[:-1])
                ansatz[j][int(ansatz[j][i][-1])] = k_p

            # if k.find('_') < 0:
            if k[0] != 'C':
                ansatz[j][i] = k
            else:
                q_p_arr = self.rng.permutation(self.n_qubits)
                for q_p in q_p_arr:
                    if q_p == i:
                        continue
                    if ansatz[j][q_p].find('_') > 0:
                        double_swap_flag += 1
                        k_pp = self.rng.choice(self.gates_arr, p=self.gates_probs)
                        if double_swap_flag == 2 and k_pp[0] == 'C':
                            direction = self.rng.permutation(['_C', '_T'])
                            ansatz[j][int(ansatz[j][i][-1])] = k + direction[0] + f'-{int(ansatz[j][q_p][-1])}'
                            ansatz[j][int(ansatz[j][q_p][-1])] = k + direction[1] + f'-{int(ansatz[j][i][-1])}'
                        else:
                            ansatz[j][int(ansatz[j][q_p][-1])] = self.rng.choice(self.gates_arr[:-1])

                    direction = self.rng.permutation(['_C', '_T'])
                    ansatz[j][i] = k + direction[0] + f'-{q_p}'
                    ansatz[j][q_p] = k + direction[1] + f'-{i}'
                    break

        self.population.append(ansatz)
