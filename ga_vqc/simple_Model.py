import copy
import datetime
import difflib
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from .Distance import euclidean_distances, tsne
from .GA_ABC import GA_Model
from .GA_Support import make_results_json
from .simple_Individual import Individual


class Model(GA_Model):
    """
    Container for all the logic of the GA Model.

    TODO: change the I assignment to a random assignment.
    """

    def __init__(self, config, set_of_preran_circuits=set()):
        """
        TODO: don't use 'self.' anymore for hyperparams, just pass around config so its clear 
            what state is set at runtime and doesnt change
        """
        ### hyperparams for GA ###
        self.backend_type = config.backend_type
        self.vqc = config.vqc
        self.max_concurrent = config.max_concurrent

        self.n_qubits = config.n_qubits
        self.max_generate_moments = config.max_moments
        self.max_vector_moments = 2 * config.max_moments
        self.add_moment_prob = config.add_moment_prob
        self.genepool = config.genepool
        self.pop_size = config.pop_size
        self.init_pop_size = config.init_pop_size
        self.n_new_individuals = config.n_new_individuals
        self.n_winners = config.n_winners
        self.n_mutations = config.n_mutations
        self.n_steps_patience = config.n_steps_patience
        self.n_eval_metrics = config.n_eval_metrics
        self.best_perf = {
            "fitness": 0,
            "eval_metrics": [],
            "ansatz": None,
            "ansatz_dicts": [],
            "ansatz_draw": str(),
            "generation": 0,
            "index": 0,
        }

        self.population = []
        self.fitness_arr = []
        self.metrics_arr = []

        self.set_of_preran_circuits = set_of_preran_circuits
        self.set_of_all_circuits = copy.deepcopy(set_of_preran_circuits)
        self.full_population = []
        self.full_fitness_arr = []
        self.full_metrics_arr = []

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Following time variables for debugging purposes only #
        self.total_ga_time = 0
        self.total_vqc_time = 0
        self.retrain_time = 0

        self.ga_output_path = config.ga_output_path
        self.rng_seed = config.rng_seed
        self.rng = np.random.default_rng(seed=self.rng_seed)

        ### hyperparams for qae ###
        self.vqc_config = config.vqc_config
        self.vqc_config["n_ansatz_qubits"] = self.n_qubits
        self.vqc_config["start_time"] = self.start_time

        self.generate_initial_pop()

    def generate_ansatz(self):
        ansatz = Individual(
                    self.n_qubits,
                    self.max_generate_moments,
                    self.max_vector_moments,
                    self.genepool,
                    self.rng_seed,
                )
        if ansatz.n_moments > self.max_vector_moments:
            self.update_circuit_set(2 * ansatz.n_moments)
        return ansatz
    
    def update_circuit_set(self, max_vector_moments):
        self.max_vector_moments = max_vector_moments
        self.set_of_all_circuits = copy.deepcopy(self.set_of_preran_circuits)
        for ansatz in self.full_population:
            ansatz.vectorize(max_vector_moments)
            self.set_of_all_circuits.add(tuple(ansatz.vector))

    def generate_initial_pop(self):
        """
        Generates the initial population by making many more individuals than needed, and pruning down to pop_size by taking maximally different individuals.

        TODO: Change to a custom distance calculation that can be stored in each ansatz?
        """
        start_time = time.time()
        init_pop = []
        for _ in range(self.init_pop_size):
            init_pop.append(
                self.generate_ansatz()
            )

        ## Euclidean Distance on Vectorized Circuits ##
        selected_ixs = set()
        for i in range(self.pop_size):
            if i == 0:
                random_ix = self.rng.integers(0, self.init_pop_size)
                self.population.append(init_pop[random_ix])
                continue

            distances_arr = []
            for j in range(len(self.population)):
                distances_arr.append(
                    np.array(
                        euclidean_distances(
                            self.population[j], 
                            init_pop,
                        )
                    )
                )

            sorted_ixs = np.argsort(np.mean(distances_arr, axis=0))
            selected_ix = sorted_ixs[-1]
            k = 1
            while selected_ix in selected_ixs:
                k += 1
                selected_ix = sorted_ixs[-k]
            selected_ixs.add(selected_ix)
            self.population.append(init_pop[selected_ix])
        
        end_time = time.time()
        exec_time = end_time - start_time
        self.total_ga_time += exec_time
        print(f"Initial generation/selection in {exec_time:.2f} seconds")

    def evolve(self):
        """
        Evolves the GA.
        """
        gen = 0
        while True:
            print(f"GA iteration {gen}")

            # Evaluating Fitness (Running VQCs) # 
            vqc_start_time = time.time()
            self.evaluate_fitness(gen)
            vqc_end_time = time.time()
            self.total_vqc_time += (vqc_end_time - vqc_start_time)

            # GA Post-processing #
            post_process_start_time = time.time()

            results = self.make_results(gen)
            
            parents = self.select()

            num_children, num_immigrants = self.anneal_manager(gen)

            children = self.mate(parents, num_children)
            immigrants = self.immigrate(num_immigrants)

            self.population.extend(children)
            self.population.extend(immigrants)

            print(
                f"Best fitness: {self.best_perf['fitness']}, " + 
                f"Best metrics: {self.best_perf['eval_metrics']}, \n" +
                f"Best ansatz: {self.best_perf['ansatz_draw']}"
            )

            
            if (gen - self.best_perf["generation"]) > self.n_steps_patience:
                break
            print(
                "filepath is: ",
                make_results_json(results, self.start_time, self.ga_output_path, gen)
            )
            gen += 1

            post_process_end_time = time.time()
            self.total_ga_time += (post_process_end_time - post_process_start_time)
            running_total_time = self.total_ga_time + self.total_vqc_time
            print(f'GA (classical) computations have taken {self.total_ga_time:0.2f} seconds \n     and {100 * self.total_ga_time / running_total_time:0.2f}% of the total time so far.')
            print(f'VQC (quantum) computations have taken {self.total_vqc_time:0.2f} seconds \n     and {100 * self.total_vqc_time / running_total_time:0.2f}% of the total time so far.')

        print(
            "filepath is: ",
            make_results_json(
                results, self.start_time, self.ga_output_path, gen, final_flag=True
            ),
        )

        post_process_end_time = time.time()
        self.total_ga_time += (post_process_end_time - post_process_start_time)

        ### Final re-training for std. dev. estimate ###
        retrain_start_time = time.time()

        ansatz = self.best_perf["ansatz"]
        vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
        vqc_config_ansatz["ansatz_dicts"] = ansatz.ansatz_dicts
        vqc_config_ansatz["ansatz_qml"] = ansatz.ansatz_qml
        vqc_config_ansatz["ansatz_draw"] = ansatz.ansatz_draw
        vqc_config_ansatz["params"] = ansatz.params
        vqc_config_ansatz["gen"] = gen
        
        output_arr = []
        for i in range(20):
            vqc_config_ansatz["ix"] = i
            output_arr.append(self.vqc(vqc_config_ansatz))
        self.fitness_arr = [output["fitness_metric"] for output in output_arr]
        print(f"Final fitness distribution: {self.fitness_arr}")
        print(f"Avg fitness: {np.mean(self.fitness_arr)},  Std Dev: {np.std(self.fitness_arr)}, Std Dev of Mean: {np.std(self.fitness_arr) / (20**0.5)}")
        if self.n_eval_metrics > 0:
            self.metrics_arr = [output["eval_metrics"] for output in output_arr]
            for metric in self.metrics_arr[0].keys():
                metric_list = [m[metric] for m in self.metrics_arr]
                print(f"Final {metric} distribution: {metric_list}")
                print(f"Avg {metric}: {np.mean(metric_list)},  Std Dev: {np.std(metric_list)}, Std Dev of Mean: {np.std(metric_list) / (20**0.5)}")

        retrain_end_time = time.time()
        self.retrain_time = retrain_end_time - retrain_start_time

        TOTAL_TIME = self.total_ga_time + self.total_vqc_time + self.retrain_time
        print(f'Final GA (classical) time: {self.total_ga_time:0.2f} seconds')
        print(f'GA fraction of total time: {100 * self.total_ga_time / TOTAL_TIME:0.2f} %')
        print(f'Final VQC (quantum) time: {self.total_vqc_time:0.2f} seconds')
        print(f'VQC fraction of total time: {100 * self.total_vqc_time / TOTAL_TIME:0.2f} %')
        print(f'Final retrain (quantum, for final statistics) time: {self.retrain_time:0.2f} seconds')
        print(f'Retrain fraction of total time: {100 * self.retrain_time / TOTAL_TIME:0.2f} %')

    def evaluate_fitness(self, gen):
        """
        Evaluates the fitness level of all ansatz. Runs the QML optimization task.

        TODO: make distinction between 'max_moments' for generating new cirsuits and 'max_moments' for making vectors
        """
        ix = 0
        args_arr = []
        for ansatz in self.population:
            self.full_population.append(copy.deepcopy(ansatz))
            self.set_of_all_circuits.add(tuple(ansatz.vector))

            vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
            vqc_config_ansatz["ansatz_dicts"] = ansatz.ansatz_dicts
            vqc_config_ansatz["ansatz_qml"] = ansatz.ansatz_qml
            vqc_config_ansatz["ansatz_draw"] = ansatz.ansatz_draw
            vqc_config_ansatz["params"] = ansatz.params
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
        
        # Both the fitness metrics and eval_metrics must be JSON serializable 
        #   -> (ie. default python classes or have custom serialization)
        self.fitness_arr = [output["fitness_metric"] for output in output_arr]
        self.full_fitness_arr.extend(copy.deepcopy(self.fitness_arr))
        if self.n_eval_metrics > 0:
            self.metrics_arr = [output["eval_metrics"] for output in output_arr]
            self.full_metrics_arr.extend(copy.deepcopy(self.metrics_arr))

        end_time = time.time()
        exec_time = end_time - start_time
        self.total_vqc_time += exec_time
        print(f"QML Optimization in {exec_time:.2f} seconds")

        if self.best_perf["fitness"] < np.amax(self.fitness_arr):
            print("!! IMPROVED PERFORMANCE !!")
            self.best_perf["fitness"] = np.amax(self.fitness_arr).item()
            chosen_index = np.argmax(self.fitness_arr).item()
            self.best_perf["ansatz"] = copy.deepcopy(
                self.population[chosen_index]
            )
            self.best_perf["ansatz_dicts"] = copy.deepcopy(
                self.population[chosen_index].ansatz_dicts
            )
            self.best_perf["ansatz_draw"] = copy.deepcopy(
                self.population[chosen_index].ansatz_draw
            )
            self.best_perf["generation"] = gen
            self.best_perf["index"] = chosen_index

            if self.n_eval_metrics > 0:
                self.best_perf["eval_metrics"] = copy.deepcopy(
                    self.metrics_arr[chosen_index]
                )

    def make_results(self, gen):
        ### tSNE clustering ###
        destdir_curves = os.path.join(self.ga_output_path, "ga_curves", "run-%s" % self.start_time)
        if not os.path.exists(destdir_curves):
            os.makedirs(destdir_curves)
        for perp in range(2, len(self.full_population)):
            data_tsne = tsne(self.population, self.max_moments, rng_seed=self.rng_seed, perplexity=perp)
            x, y = data_tsne[:, 0].T, data_tsne[:, 1].T
            filepath_tsne = os.path.join(
                destdir_curves,
                "%03dtsne%04d_distance_data.png"
                % (gen, perp)
            )
            plt.figure(1)
            plt.style.use("seaborn")
            plt.scatter(x, y, marker=".", c=[m["auroc"] for m in self.full_metrics_arr], cmap=plt.set_cmap('plasma'))
            plt.ylabel("a.u.")
            plt.xlabel("a.u.")
            cbar = plt.colorbar()
            cbar.set_label("AUROC")
            plt.title("tSNE of Full Population")
            plt.savefig(filepath_tsne, format="png")
            plt.close(1)

        results = {
            "full_population_vectors": [ansatz.vector for ansatz in self.full_population],
            "full_population_fitness": self.full_fitness_arr,
            "full_generation": [i.ansatz_dicts for i in self.population],
            "full_drawn_generation": [i.ansatz_draw for i in self.population],
            "full_fitness_generation": [i for i in self.fitness_arr],
            "fitness_stats": f"Avg fitness: {np.mean(self.fitness_arr)}, Std. Dev: {np.std(self.fitness_arr)}",
            "best_ansatz": self.best_perf["ansatz_dicts"],
            "best_drawn_ansatz": self.best_perf["ansatz_draw"],
            "best_fitness": self.best_perf["fitness"],
            "best_fitness_gen": self.best_perf["generation"],
            "best_fitness_ix": self.best_perf["index"],
        }
        if self.n_eval_metrics > 0:
            results["full_population_metrics"] = self.full_metrics_arr
            results["eval_metrics"] = self.metrics_arr
            results["eval_metrics_stats"] = [
                f"Avg {k}: {np.mean([i[k] for i in self.metrics_arr])}, "
                + f"Std. Dev: {np.std([i[k] for i in self.metrics_arr])}"
                for k in self.metrics_arr[0].keys()
            ]
            results["best_eval_metrics"] = self.best_perf["eval_metrics"]

        return results

    def select(self):
        """
        Picks the top performing ansatz from a generation to mate and mutate for the next generation.
        """
        winner_arr = []
        for _ in range(self.n_winners):
            winner_ix = np.argmax(self.fitness_arr)
            winner = self.population[winner_ix]
            winner_arr.append(winner)
            self.fitness_arr.pop(winner_ix)

        self.population = []
        return winner_arr

    def mate(self, parents, num_children):
        """
        Swaps the qubits of ansatz.

        TODO: Change so that mating happens at Individual level by passing in the set of qubits that we want to swap to a function in the Individual class.
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
                    {
                        "parent_A": (j * self.n_winners) + i, 
                        "parent_B": (j * self.n_winners) + i + 1
                    }
                )

        # Perform the swap with neighboring parents
        swap_set = set()
        for swap_ix in swap_ixs:  # NOT setup for odd number of parents
            child_A = copy.deepcopy(parents[swap_ix["parent_A"]])
            child_B = copy.deepcopy(parents[swap_ix["parent_B"]])

            moment_A = self.rng.integers(child_A.n_moments) 
            moment_B = self.rng.integers(child_B.n_moments)
            qubit_A = qubit_B = self.rng.integers(self.n_qubits).item()
            swap_set.add(qubit_A)

            def check_gate(qubit_str, qubit_num):
                if qubit_str.find("_") > 0:
                    return int(qubit_str[-1])
                return qubit_num

            while True:
                A_link = check_gate(
                    child_A[moment_A, qubit_A],
                    qubit_A
                )
                B_link = check_gate(
                    child_B[moment_B, qubit_B],
                    qubit_B
                )

                if (A_link == qubit_A and B_link == qubit_B) or (A_link in swap_set and B_link in swap_set):
                    break
                else:
                    if A_link != qubit_A:
                        swap_set.add(A_link)
                        qubit_B = A_link
                    if B_link != qubit_B:
                        swap_set.add(B_link)
                        qubit_A = B_link

            child_A.overwrite(moment_A, swap_set, parents[swap_ix["parent_B"]][moment_B])
            child_B.overwrite(moment_B, swap_set, parents[swap_ix["parent_A"]][moment_A])
            if child_A.n_vector_moments > self.max_vector_moments or child_B.n_vector_moments > self.max_vector_moments:
                self.update_circuit_set(max(child_A.n_vector_moments, child_B.n_vector_moments))
            children_arr.extend([child_A, child_B])

        for child in children_arr:
            self.mutate(child)

        return children_arr[:num_children]

    def deep_permutation(self, arr):
        arr_copy = [i for i in arr]
        ix_arr = self.rng.permutation(len(arr))
        for i in range(len(arr)):
            arr[i] = arr_copy[ix_arr[i]]

        return arr

    def mutate(self, ansatz, n_mutations=None):
        """
        Mutates a single ansatz by modifying n_mutations random qubit(s) at 1 random time each.

        TODO: add in functionality to add or remove moments from an ansatz

        Variables
            j: selected moment
            i: selected qubit
            k: selected gate
        """
        ansatz_backup = copy.deepcopy(ansatz)
        if n_mutations is None:
            n_mutations = self.n_mutations
        count = 0
        while True:
            if count == 5:
                n_mutations += 1
                count = 0
            for _ in range(n_mutations):
                moment = self.rng.integers(ansatz.n_moments)
                qubit = self.rng.integers(self.n_qubits)
                ansatz.mutate(moment, qubit)

            if ansatz.n_vector_moments > self.max_vector_moments:
                self.update_circuit_set(ansatz.n_vector_moments)
            if tuple(ansatz.vector) in self.set_of_all_circuits:
                ansatz = copy.deepcopy(ansatz_backup)
                count += 1
                continue
            break

    def immigrate(self, n_individuals):
        """
        Adds in new individuals with every generation, in order to keep up the overall population diversity.
        """
        immigrant_arr = []
        for _ in range(n_individuals):
            ansatz_arr = []
            distances_arr = []
            for __ in range(100):
                ansatz = self.generate_ansatz()
                count = 0
                while True:
                    if count == 10:
                        self.max_generate_moments += 1
                        ansatz = self.generate_ansatz()
                        count = 0
                    if tuple(ansatz.vector) in self.set_of_all_circuits:
                        ansatz = self.generate_ansatz()
                        count += 1
                        continue

                    ansatz_arr.append(ansatz)
                    break
            for j in range(100):
                distances_arr.append(
                    np.mean(
                        euclidean_distances(
                            ansatz_arr[j], 
                            self.full_population
                        )
                    )
                )
            ansatz = ansatz_arr[np.argmax(distances_arr)]
            immigrant_arr.append(
                ansatz
            )

        return immigrant_arr
            
    def anneal_manager(self, gen):
        num_children = self.pop_size // 2
        num_immigrants = self.pop_size - num_children
        
        return num_children, num_immigrants
