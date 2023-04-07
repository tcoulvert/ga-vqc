import copy
import datetime
import difflib
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from .Distance import euclidian_distances, tsne
from .GA_ABC import GA_Model
from .GA_Support import make_results_json
from .simple_Individual import Individual


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
        self.genepool = config["genepool"]
        self.pop_size = config["pop_size"]
        self.init_pop_size = config["init_pop_size"]
        self.n_new_individuals = config["n_new_individuals"]
        self.n_winners = config["n_winners"]
        self.n_mutations = config["n_mutations"]
        self.n_mate_swaps = config["n_mate_swaps"]
        self.n_steps_patience = config["n_steps_patience"]
        self.best_perf = {
            "fitness": 0,
            "eval_metrics": [],
            "ansatz_dicts": [],
            "ansatz_draw": str(),
            "generation": 0,
            "index": 0,
        }

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ga_output_path = config["ga_output_path"]
        self.rng_seed = config["rng_seed"]
        self.rng = np.random.default_rng(seed=self.rng_seed)

        ### hyperparams for qae ###
        self.vqc_config = config["vqc_config"]
        self.vqc_config["n_ansatz_qubits"] = self.n_qubits
        self.vqc_config["start_time"] = self.start_time

        self.population = []
        self.fitness_arr = [0 for _ in range(self.pop_size)]
        self.metrics_arr = [dict() for _ in range(self.pop_size)]
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
                    # self.rng.integers(1, self.max_moments + 1),
                    self.max_moments,
                    self.genepool,
                    self.rng_seed,
                )
            )

        seq_mat = difflib.SequenceMatcher(isjunk=lambda x: x in " -")
        compare_arr = ["" for qubit in range(self.n_qubits)]
        distances_arr = []
        selected_ixs = set()
        for _ in range(self.pop_size):
            distances = []
            for j, individual in enumerate(init_pop):
                if j in selected_ixs:
                    distances.append(0)
                    continue
                dist = 0.0
                for qubit in range(self.n_qubits):
                    seq_mat.set_seq2(compare_arr[qubit])
                    seq_mat.set_seq1(individual.ansatz_draw[qubit])
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
            self.fitness_arr = [0 for i in self.population]
            self.evaluate_fitness(step)

            results = self.make_results()

            parents = self.select()
            self.mate(parents)
            self.immigrate()
            self.check_max_moments()

            print(
                f"Best fitness: {self.best_perf['fitness']}, " + 
                f"Best metrics: {self.best_perf['eval_metrics']}, " +
                f"Best ansatz: {self.best_perf['ansatz_dicts']}"
            )

            if step > 20:
                if (step - self.best_perf["generation"]) > self.n_steps_patience:
                    break
            make_results_json(results, self.start_time, self.ga_output_path, step)
            step += 1
        print(
            "filename is: ",
            make_results_json(
                results, self.start_time, self.ga_output_path, step, final_flag=True
            ),
        )

    def evaluate_fitness(self, gen):
        """
        Evaluates the fitness level of all ansatz. Runs the QML optimization task.

        TODO: change to do per given ansatz (so we don't have to train every ansatz).
            -> make so fitness_arr can be shorter than population
        """
        ix = 0
        args_arr = []
        for ansatz in self.population:
            ansatz.convert_to_qml()
            ansatz.draw_ansatz()
            vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
            vqc_config_ansatz["ansatz_dicts"] = ansatz.ansatz_dicts
            vqc_config_ansatz["ansatz_qml"] = ansatz.ansatz_qml
            vqc_config_ansatz["n_params"] = ansatz.n_params
            vqc_config_ansatz["ix"] = ix
            vqc_config_ansatz["gen"] = gen
            args_arr.append(copy.deepcopy(vqc_config_ansatz))
            ix += 1

        start_time = time.time()
        output_arr = []
        for i in range(self.pop_size // self.max_concurrent):
            with mp.get_context("spawn").Pool(processes=self.max_concurrent) as pool:
                output_arr.extend(
                    pool.map(
                        self.vqc,
                        args_arr[
                            i * self.max_concurrent : (i + 1) * self.max_concurrent
                        ],
                    )
                )
        for i in range(len(output_arr)):
            # Both the fitness metrics and eval_metrics must be JSON serializable -> (ie. 
            #  default python classes or have custom serialization)
            self.fitness_arr[i] = output_arr[i]["fitness_metric"]
            self.metrics_arr[i] = output_arr[i]["eval_metrics"]
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"QML Optimization in {exec_time:.2f} seconds")

        if self.best_perf["fitness"] < np.amax(self.fitness_arr):
            print("!! IMPROVED PERFORMANCE !!")
            self.best_perf["fitness"] = np.amax(self.fitness_arr).item()
            self.best_perf["eval_metrics"] = copy.deepcopy(
                self.metrics_arr[np.argmax(self.fitness_arr)]
            )
            self.best_perf["ansatz_dicts"] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)].ansatz_dicts
            )
            self.best_perf["ansatz_draw"] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)].ansatz_draw
            )
            self.best_perf["generation"] = gen
            self.best_perf["index"] = np.argmax(self.fitness_arr).item()

    def make_results(self):

        distances_from_best = euclidian_distances(self.best_perf["ansatz_dicts"], self.population)
        destdir_curves = os.path.join(self.ga_output_path, "ga_curves")
        if not os.path.exists(destdir_curves):
            os.makedirs(destdir_curves)
        filepath_euclid = os.path.join(
            destdir_curves,
            "%03deuclid_distance-%d_data.png"
            % (self.best_perf["generation"], self.start_time),
        )
        plt.figure(0)
        plt.style.use("seaborn")
        plt.scatter(distances_from_best, self.fitness_arr, marker=".", color="g")
        plt.ylabel("Fitness")
        plt.xlabel("Euclidian distance from best ansatz")
        plt.savefig(filepath_euclid, format="png")
        # data_tsne = tsne(self.population)
        # filepath_tsne = os.path.join(
        #     destdir_curves,
        #     "%03dtsne_distance-%d_data.png"
        #     % (self.best_perf["generation"], self.start_time),
        # )
        # plt.figure(0)
        # plt.style.use("seaborn")
        # plt.scatter([i[0] for i in data_tsne.T], [i[1] for i in data_tsne.T], marker=".", cmap=)
        # plt.ylabel("a.u.")
        # plt.xlabel("a.u.")
        # plt.savefig(filepath_tsne, format="png")

        results = {
            "full_population": [i.ansatz_dicts for i in self.population],
            "full_drawn_population": [i.ansatz_draw for i in self.population],
            "full_fitness": self.fitness_arr,
            "fitness_stats": f"Avg fitness: {np.mean(self.fitness_arr)}, Std. Dev: {np.std(self.fitness_arr)}",
            "full_eval_metrics": self.metrics_arr,
            "eval_metrics_stats": [
                f"Avg {k}: {np.mean([i[k] for i in self.metrics_arr])}, "
                + f"Std. Dev: {np.std([i[k] for i in self.metrics_arr])}"
                for k in self.metrics_arr[0].keys()
            ],
            "full_distances": distances_from_best,
            "distances_stats": f"Avg distance: {np.mean(distances_from_best)}, Std. Dev: {np.std(distances_from_best)}",
            "best_ansatz": self.best_perf["ansatz_dicts"],
            "best_drawn_ansatz": self.best_perf["ansatz_draw"],
            "best_fitness": self.best_perf["fitness"],
            "best_eval_metrics": self.best_perf["eval_metrics"],
            "best_fitness_gen": self.best_perf["generation"],
            "best_fitness_ix": self.best_perf["index"],
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
        swap_set = set()
        for swap_ix in swap_ixs:  # set up for odd number of parents
            children = [copy.deepcopy(parents[swap_ix[0]]), copy.deepcopy(parents[swap_ix[1]])]
            moment_A = self.rng.integers(children[0].n_moments) 
            moment_B = self.rng.integers(children[1].n_moments)
            # children = {
            #     "child_A": copy.deepcopy(parents[swap_ix[0]]), 
            #     "child_B": copy.deepcopy(parents[swap_ix[1]])
            # }
            # print(f"Pre-mateswap ansatz: {children}")
            # moment_A = self.rng.integers(children["child_A"].n_moments) 
            # moment_B = self.rng.integers(children["child_B"].n_moments)
            # qubit_A = qubit_B = self.rng.integers(self.n_qubits).item()
            # swap_set.add(qubit_A)


            # def check_gate(qubit_str):
            #     if qubit_str.find("_") > 0:
            #         return int(qubit_str[-1])
            #     return -1


            # # USE RECURSION !!!!!!!!
            # while True:
            #     qubit_B_new = check_gate(
            #         children["child_A"][moment_A, qubit_A]
            #     )
            #     qubit_A_new = check_gate(
            #         children["child_B"][moment_B, qubit_B]
            #     )

            #     if qubit_A_new == -1:
            #         if qubit_B_new == -1:
            #             break
            #         # need to check moment_B, qubit_A_new

            #     if qubit_A_new == qubit_B_new:
            #         swap_set.add(qubit_A_new)
            #         break

            #     i0_new = i1_new = -1
            #     if children[0][moment_A, qubit_A].find("_") > 0:
            #         i1_new = int(children[0][moment_A, qubit_A][-1])
            #         swap_set.add(i1_new)
            #     if children[1][moment_B, qubit_B].find("_") > 0:
            #         i0_new = int(children[1][moment_B, qubit_B][-1])
            #         if i0_new == i1_new:
            #             break
            #         swap_set.add(i0_new)
            #     qubit_A, qubit_B = i0_new, i1_new

            #     if qubit_A < 0 or qubit_B < 0:
            #         break

            # for qubit in swap_set:
            #     children[0][moment_A, qubit] = parents[swap_ix[1]][moment_B, qubit]
            #     children[1][moment_B, qubit] = parents[swap_ix[0]][moment_A, qubit]

            # print(f"Post-mateswap ansatz: {children}")
            # ADDED IN FOR MOMENT SWAP
            # children["child_A"][moment_A] = parents[swap_ix[1]][moment_B]
            # children["child_B"][moment_B] = parents[swap_ix[0]][moment_A]
            children[0][moment_A] = parents[swap_ix[1]][moment_B]
            children[1][moment_B] = parents[swap_ix[0]][moment_A]
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

        Variables
            j: selected moment
            i: selected qubit
            k: selected gate
        """
        for _ in range(self.n_mutations):
            if (
                ansatz.n_moments < self.max_moments
                and self.rng.random() < self.add_moment_prob
            ):
                ansatz.add_moment(method='duplicate')
            
            moment = self.rng.integers(ansatz.n_moments)
            qubit = self.rng.integers(self.n_qubits)
            ansatz.mutate(moment, qubit)

        self.population.append(ansatz)

    def immigrate(self):
        """
        Adds in new individuals with every generation, in order to keep up the overall population diversity.
        """
        for _ in range(self.n_new_individuals):
            self.population.append(
                Individual(
                    self.n_qubits,
                    self.rng.integers(1, self.max_moments + 1),
                    self.genepool,
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
