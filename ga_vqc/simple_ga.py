import copy
import datetime
import difflib
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import pennylane as qml
import psutil
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from .GA_ABC import GA_Individual, GA_Model
from .GA_Support import make_results_json


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
        self.ansatz_draw = []
        self.params = []

        self.generate()  # Should this be done automatically?
        self.convert_to_qml()
        self.draw_ansatz()

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
            ix = self.rng.permutation(self.n_qubits)  # which qubit to pick first
            for i in ix:
                if self.ansatz[j][i] != 0:
                    continue
                k = self.rng.choice(self.gates_arr, p=self.gates_probs)

                # if k.find('_') < 0:
                if k[0] != "C":
                    self.ansatz[j][i] = k
                    continue

                q_p_arr = self.rng.permutation(
                    self.n_qubits
                )  # qubit_pair_array for 2-qubit gates
                for q_p in q_p_arr:
                    if self.ansatz[j][q_p] != 0 or q_p == i:
                        continue

                    direction = self.rng.permutation(["_C", "_T"])
                    self.ansatz[j][i] = k + direction[0] + f"-{q_p}"
                    self.ansatz[j][q_p] = k + direction[1] + f"-{i}"
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
        self.ansatz_qml = []
        self.params = []
        for j in range(len(self.ansatz)):
            moment_dict = {i: list() for i in self.gates_arr}
            stored_i = []
            for i in range(self.n_qubits):
                _ix = self.ansatz[j][i].find("_")
                if _ix < 0:
                    moment_dict[self.ansatz[j][i]].append(i)
                else:
                    if i in stored_i:
                        continue
                    _1ix = self.ansatz[j][i][_ix + 1]
                    q_p = int(self.ansatz[j][i][-1])
                    stored_i.append(q_p)
                    if _1ix == "C":
                        moment_dict[self.ansatz[j][i][:_ix]].append([i, int(q_p)])
                    else:
                        moment_dict[self.ansatz[j][i][:_ix]].append([int(q_p), i])

            for k in moment_dict.keys():
                if len(moment_dict[k]) == 0 or k == "I":  # add in identity gates?
                    continue
                if len(k) <= 2:
                    self.ansatz_qml.append(
                        f"qml.broadcast(qml.{k}, wires={moment_dict[k]}, pattern='single', parameters=params[{len(self.params)}:{len(self.params)+len(moment_dict[k])}])"
                    )
                    # change to allow for one-qubit gates with 0 or 2+ params
                    for i in range(len(moment_dict[k])):
                        self.params.append(0.0)
                elif (
                    len(k) > 2
                ):  # Assumes the 2-qubit gates have no parameters, which is not generally true
                    self.ansatz_qml.append(
                        f"qml.broadcast(qml.{k}, wires={np.array(moment_dict[k]).flatten(order='C').tolist()}, pattern={moment_dict[k]})"
                    )
                    # change to allow for two-qubit gates with 1+ params

    def ansatz_circuit(self, params, event=None, ansatz=None):
        for m in ansatz:
            # exec(m)
            try:
                exec(m)
            except:
                print(m)
                raise Exception("oopsies there's a problem")
        return qml.expval(qml.PauliZ(wires=[self.n_qubits - 1]))

    def draw_ansatz(self):
        self.ansatz_draw = []
        full_ansatz_draw = qml.draw(
            qml.QNode(
                self.ansatz_circuit,
                qml.device("default.qubit", wires=self.n_qubits, shots=1),
            ),
            decimals=None,
            expansion_strategy="device",
            show_all_wires=True,
        )(self.params, event=[i for i in range(self.n_qubits)], ansatz=self.ansatz_qml)[
            :-3
        ]
        indices = [i for i, c in enumerate(full_ansatz_draw) if c == ":"]
        for _ in range(self.n_qubits):
            self.ansatz_draw.append(full_ansatz_draw[: indices[1] - 2][3:-2])
            full_ansatz_draw = full_ansatz_draw[indices[1] - 1 :]

    def add_moment(self):
        j = self.rng.integers(self.n_moments)
        self.ansatz.append(copy.deepcopy(self.ansatz[j]))
        self.n_moments += 1


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
        self.backend_type = config["backend_type"]
        self.vqc = config["vqc"]
        self.max_concurrent = config["max_concurrent"]

        self.n_qubits = config["n_qubits"]
        self.max_moments = config["max_moments"]
        self.add_moment_prob = config["add_moment_prob"]
        self.gates_arr = config["gates_arr"]
        self.gates_probs = config["gates_probs"]
        self.pop_size = config["pop_size"]
        self.init_pop_size = config["init_pop_size"]
        self.n_new_individuals = config["n_new_individuals"]
        self.n_winners = config["n_winners"]
        self.n_mutations = config["n_mutations"]
        self.n_mate_swaps = config["n_mate_swaps"]
        self.n_steps = config["n_steps"]
        self.best_perf = [0, [], 0, str(), 0]  # change to dict

        ### hyperparams for qae ###
        self.latent_qubits = config["latent_qubits"]
        self.n_shots = config["n_shots"]
        self.events = config["events"]
        self.train_size = config["train_size"]
        self.batch_size = config["batch_size"]

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.script_path = os.path.dirname(os.path.realpath(__file__))
        self.rng_seed = config["seed"]
        self.rng = np.random.default_rng(seed=self.rng_seed)

        self.population = []
        self.fitness_arr = [0 for i in range(self.pop_size)]
        self.generate_initial_pop()

    def generate_initial_pop(self):
        """
        Generates the initial population by making many more individuals than needed, and pruning down to pop_size by taking maximally different individuals.

        TODO: Change to a custom distance calculation that can be stored in each ansatz?
        """
        start_time = time.time()
        init_pop = []
        for _ in range(self.init_pop_size):
            init_pop.append(
                Individual(
                    self.n_qubits,
                    self.rng.integers(1, self.max_moments),
                    self.gates_arr,
                    self.gates_probs,
                    self.rng_seed,
                )
            )

        seq_mat = difflib.SequenceMatcher(isjunk=lambda x: x in " -")
        compare_arr = ["" for i in range(self.n_qubits)]
        distances_arr = []
        selected_ixs = set()
        for _ in range(self.pop_size):
            distances = []
            for j, individual in enumerate(init_pop):
                if j in selected_ixs:
                    distances.append(0)
                    continue
                dist = 0.0
                for i in range(self.n_qubits):
                    seq_mat.set_seq2(compare_arr[i])
                    seq_mat.set_seq1(individual.ansatz_draw[i])
                    dist += 1 - seq_mat.ratio()
                distances.append(dist / self.n_qubits)

            distances_arr.append(np.array(distances))
            selected_ix = np.argmax(np.mean(distances_arr, axis=0))
            selected_ixs.add(selected_ix)
            self.population.append(init_pop[selected_ix])
            compare_arr = init_pop[selected_ix].ansatz_draw
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"Initial generation/selection in {exec_time:.2f} seconds")

    def evolve(self):
        """
        Evolves the GA.
        """
        step = 0
        while True:
            print(f"GA iteration {step}")
            print(
                f"Mem GA process - {psutil.Process().memory_info().rss / (1024 * 1024)}"
            )
            self.fitness_arr = [0 for i in self.population]
            self.evaluate_fitness(step)

            results = self.make_results()

            parents = self.select()
            self.mate(parents)
            self.immigrate()
            self.check_max_moments()

            print(
                f"Best Fitness: {self.best_perf[0]}, Best ansatz: {self.best_perf[1]}"
            )

            if step > 20:
                if (step - self.best_perf[2]) > self.n_steps:
                    break
            make_results_json(results, self.start_time, self.script_path, step)
            step += 1
        print(
            "filename is: ",
            make_results_json(
                results, self.start_time, self.script_path, step, final_flag=True
            ),
        )

    def evaluate_fitness(self, gen):
        """
        Evaluates the fitness level of all ansatz. Runs the QML optimization task.

        TODO: change to do per given ansatz (so we don't have to train every ansatz).
            -> make so fitness_arr can be shorter than population
        """
        self.fitness_arr = []
        ix = 0
        args_arr = []
        for p in self.population:
            p.convert_to_qml()
            p.draw_ansatz()
            event_sub = self.rng.choice(self.events, self.train_size, replace=False)
            args_arr.append(
                (
                    p.ansatz_qml,
                    p.ansatz,
                    p.params,
                    event_sub,
                    self.train_size,
                    self.batch_size,
                    self.n_qubits,
                    self.latent_qubits,
                    self.rng_seed,
                    ix,
                    gen,
                    self.start_time,
                    self.n_shots,
                )
            )
            ix += 1

        start_time = time.time()
        for i in range(self.pop_size // self.max_concurrent):
            with mp.get_context("spawn").Pool(processes=len(args_arr)) as pool:
                self.fitness_arr.extend(
                    pool.starmap(
                        self.vqc,
                        args_arr[
                            i * self.max_concurrent : (i + 1) * self.max_concurrent
                        ],
                    )
                )
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"QML Optimization in {exec_time:.2f} seconds")

        if self.best_perf[0] < np.amax(self.fitness_arr):
            print("!! IMPROVED PERFORMANCE !!")
            self.best_perf[0] = np.amax(self.fitness_arr)
            self.best_perf[1] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)].ansatz
            )
            self.best_perf[3] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)].ansatz_draw
            )
            self.best_perf[2] = gen
            self.best_perf[4] = np.argmax(self.fitness_arr)

    def make_results(self):
        results = {
            "full_population": [i.ansatz for i in self.population],
            "full_drawn_population": [i.ansatz_draw for i in self.population],
            "full_fitness": [i.tolist() for i in self.fitness_arr],
            "fitness_stats": f"Avg fitness: {np.mean(self.fitness_arr)}, Std. Dev: {np.std(self.fitness_arr)}",
            "best_ansatz": self.best_perf[1],
            "best_drawn_ansatz": self.best_perf[3],
            "best_fitness": self.best_perf[0].item(),
            "best_fitness_gen": self.best_perf[2],
            "best_fitness_ix": self.best_perf[4].item(),
        }
        return results

    def select(self):
        """
        Picks the top performing ansatz from a generation to mate and mutate for the next generation.
        """
        winner_arr = []
        for i in range(self.n_winners):
            winner_ix = np.argmax(self.fitness_arr)
            winner = self.population[winner_ix]
            winner_arr.append(winner)
            self.fitness_arr.pop(winner_ix)

        self.population = []
        return winner_arr

    def mate(self, parents):
        """
        Swaps the qubits of ansatz.

        TODO:
        """
        children_arr = []
        swap_ixs = []
        parents = self.deep_permutation(parents)
        while len(parents) < self.pop_size - self.n_new_individuals:
            parents.extend(self.deep_permutation(parents))

        # Create index pairings for swapping
        for j in range(len(parents) // self.n_winners):
            for i in range(self.n_winners):
                if i % 2 != 0:
                    continue
                swap_ixs.append(
                    [(j * self.n_winners) + i, (j * self.n_winners) + i + 1]
                )

        # Perform the swap with neighboring parents
        i_set = set()
        for swap_ix in swap_ixs:  # set up for odd number of parents
            children = [parents[swap_ix[0]], parents[swap_ix[1]]]
            j0, j1 = self.rng.integers(children[0].n_moments), self.rng.integers(
                children[1].n_moments
            )
            i0 = i1 = self.rng.integers(self.n_qubits)

            i_set.add(i0)
            while True:
                i0_new = i1_new = -1
                if children[0][j0, i0].find("_") > 0:
                    i1_new = int(children[0][j0, i0][-1])
                    i_set.add(i1_new)
                if children[1][j1, i1].find("_") > 0:
                    i0_new = int(children[1][j1, i1][-1])
                    if i0_new == i1_new:
                        break
                    i_set.add(i0_new)
                i0, i1 = i0_new, i1_new

                if i0 < 0 or i1 < 0:
                    break

            for i in i_set:
                children[0][j0, i] = parents[swap_ix[1]][j1, i]
                children[1][j1, i] = parents[swap_ix[0]][j0, i]

            children_arr.extend(children)

        for child in children_arr:
            if len(self.population) < self.pop_size - self.n_new_individuals:
                self.mutate(child)

    def deep_permutation(self, arr):
        arr_copy = [i for i in arr]
        ix_arr = self.rng.permutation(len(arr))
        for i in range(len(arr)):
            arr[i] = arr_copy[ix_arr[i]]

        return arr

    def mutate(self, ansatz):
        """
        Mutates a single ansatz by modifying n_mutations random qubit(s) at 1 random time each.

        TODO: add in functionality to add or remove moments from an ansatz
        """
        for _ in range(self.n_mutations):
            double_swap_flag = 0
            if (
                ansatz.n_moments < self.max_moments
                and self.rng.random() < self.add_moment_prob
            ):
                ansatz.add_moment()
                j = ansatz.n_moments - 1
            else:
                j = self.rng.integers(ansatz.n_moments)
            i = self.rng.integers(self.n_qubits)
            k = self.rng.choice(self.gates_arr, p=self.gates_probs)

            if ansatz[j][i].find("_") > 0:
                double_swap_flag += 1
                k_p = self.rng.choice(self.gates_arr[:-1])
                ansatz[j][int(ansatz[j][i][-1])] = k_p

            # if k.find('_') < 0:
            if k[0] != "C":
                ansatz[j][i] = k
            else:
                q_p_arr = self.rng.permutation(self.n_qubits)
                for q_p in q_p_arr:
                    if q_p == i:
                        continue
                    if ansatz[j][q_p].find("_") > 0:
                        double_swap_flag += 1
                        k_pp = self.rng.choice(self.gates_arr, p=self.gates_probs)
                        if double_swap_flag == 2 and k_pp[0] == "C":
                            direction = self.rng.permutation(["_C", "_T"])
                            ansatz[j][int(ansatz[j][i][-1])] = (
                                k + direction[0] + f"-{int(ansatz[j][q_p][-1])}"
                            )
                            ansatz[j][int(ansatz[j][q_p][-1])] = (
                                k + direction[1] + f"-{int(ansatz[j][i][-1])}"
                            )
                        else:
                            ansatz[j][int(ansatz[j][q_p][-1])] = self.rng.choice(
                                self.gates_arr[:-1]
                            )

                    direction = self.rng.permutation(["_C", "_T"])
                    ansatz[j][i] = k + direction[0] + f"-{q_p}"
                    ansatz[j][q_p] = k + direction[1] + f"-{i}"
                    break

        self.population.append(ansatz)

    def immigrate(self):
        """
        Adds in new individuals with every generation, in order to keep up the overall population diversity.
        """
        for _ in range(self.n_new_individuals):
            self.population.append(
                Individual(
                    self.n_qubits,
                    self.rng.integers(1, self.max_moments),
                    self.gates_arr,
                    self.gates_probs,
                    self.rng_seed,
                )
            )

    def check_max_moments(self):
        """
        Checks if many of the ansatz are 'full' with respect to max_moments, and if so it raises the ceiling
        """
        count = 0
        for ansatz in self.population:
            if ansatz.n_moments == self.max_moments:
                count += 1
        if count / self.pop_size > 0.8:
            self.max_moments += 10
