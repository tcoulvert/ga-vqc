import copy
import datetime
import difflib
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from .Distance import euclidean_distances, tsne, create_vector
from .GA_ABC import GA_Model
from .GA_Support import make_results_json
from .simple_Individual import Individual


class Model(GA_Model):
    """
    Container for all the logic of the GA Model.

    TODO: change the I assignment to a random assignment.
    """

    def __init__(self, config):
        """
        TODO: don't use 'self.' anymore for hyperparams, just pass around config so its clear 
            what state is set at runtime and doesnt change
        """
        ### hyperparams for GA ###
        self.backend_type = config.backend_type
        self.vqc = config.vqc
        self.max_concurrent = config.max_concurrent

        self.n_qubits = config.n_qubits
        self.max_moments = config.max_moments
        self.add_moment_prob = config.add_moment_prob
        self.genepool = config.genepool
        self.pop_size = config.pop_size
        self.init_pop_size = config.init_pop_size
        self.n_new_individuals = config.n_new_individuals
        self.n_winners = config.n_winners
        self.n_mutations = config.n_mutations
        self.n_steps_patience = config.n_steps_patience
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
        self.set_of_all_circuits = set()
        self.fitness_arr = [0 for _ in range(self.pop_size)]
        self.metrics_arr = [dict() for _ in range(self.pop_size)]

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
                    self.max_moments,
                    self.genepool,
                    self.rng_seed,
                )
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
                            max_moments=self.max_moments
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
            fitness_arr_gen_start_time = time.time()

            self.fitness_arr = [0 for i in self.population]

            fitness_arr_gen_end_time = time.time()
            self.total_ga_time += (fitness_arr_gen_end_time - fitness_arr_gen_start_time)

            vqc_start_time = time.time()

            self.evaluate_fitness(gen)

            vqc_end_time = time.time()
            self.total_vqc_time += (vqc_end_time - vqc_start_time)

            post_process_start_time = time.time()

            results = self.make_results(gen)

            parents = self.select()

            num_children, num_immigrants = self.anneal_manager(;fn)

            children = self.mate(parents, num_children)
            immigrants = self.immigrate(num_immigrants)

            self.population.extend(children, immigrants)
            # self.check_max_moments()

            print(
                f"Best fitness: {self.best_perf['fitness']}, " + 
                f"Best metrics: {self.best_perf['eval_metrics']}, " +
                f"Best ansatz: {self.best_perf['ansatz_dicts']}"
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

        print(
            "filepath is: ",
            make_results_json(
                results, self.start_time, self.ga_output_path, gen, final_flag=True
            ),
        )

        post_process_end_time = time.time()
        self.total_ga_time += (post_process_end_time - post_process_start_time)

        retrain_start_time = time.time()

        ### Final re-training for std. dev. estimate ###
        ansatz = self.best_perf["ansatz"]
        vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
        vqc_config_ansatz["ansatz_dicts"] = ansatz.ansatz_dicts
        vqc_config_ansatz["ansatz_qml"] = ansatz.ansatz_qml
        vqc_config_ansatz["params"] = ansatz.params
        vqc_config_ansatz["gen"] = gen
        fitness_arr = []
        auroc_arr = []
        for i in range(20):
            vqc_config_ansatz["ix"] = i
            output_dict = self.vqc(vqc_config_ansatz)
            fitness_arr.append(output_dict["fitness_metric"])
            auroc_arr.append(output_dict["eval_metrics"]["auroc"])
        print(f"Final fitness distribution: {fitness_arr}")
        print(f"Avg fitness: {np.mean(fitness_arr)},  Std Dev: {np.std(fitness_arr)}, Std Dev of Mean: {np.std(fitness_arr) / (20**0.5)}")
        print(f"Final AUROC distribution: {auroc_arr}")
        print(f"Avg AUROC: {np.mean(auroc_arr)},  Std Dev: {np.std(auroc_arr)}, Std Dev of Mean: {np.std(auroc_arr) / (20**0.5)}")

        retrain_end_time = time.time()
        self.retrain_time = retrain_end_time - retrain_start_time

        TOTAL_TIME = self.total_ga_time + self.total_vqc_time + self.retrain_time
        print(f'Final GA (classical) time: {self.total_ga_time} seconds')
        print(f'GA fraction of total time: {self.total_ga_time / TOTAL_TIME} %')
        print(f'Final VQC (quantum) time: {self.total_vqc_time} seconds')
        print(f'VQC fraction of total time: {self.total_vqc_time / TOTAL_TIME} %')
        print(f'Final retrain (quantum, for final statistics) time: {self.retrain_time} seconds')
        print(f'Retrain fraction of total time: {self.retrain_time / TOTAL_TIME} %')

    def evaluate_fitness(self, gen):
        """
        Evaluates the fitness level of all ansatz. Runs the QML optimization task.

        TODO: 1. change to do per given ansatz (so we don't have to train every ansatz).
            -> make so fitness_arr can be shorter than population
            2. check if the set works???
        """
        ix = 0
        args_arr = []
        for ansatz in self.population:
            self.set_of_all_circuits.add(tuple(create_vector(ansatz, return_type='list')))

            vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
            vqc_config_ansatz["ansatz_dicts"] = ansatz.ansatz_dicts
            vqc_config_ansatz["ansatz_qml"] = ansatz.ansatz_qml
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
        for i in range(len(output_arr)):
            # Both the fitness metrics and eval_metrics must be JSON serializable -> (ie. 
            #  default python classes or have custom serialization)
            self.fitness_arr[i] = output_arr[i]["fitness_metric"]
            self.metrics_arr[i] = output_arr[i]["eval_metrics"]

        end_time = time.time()
        exec_time = end_time - start_time
        self.total_vqc_time += exec_time
        print(f"QML Optimization in {exec_time:.2f} seconds")

        if self.best_perf["fitness"] < np.amax(self.fitness_arr):
            print("!! IMPROVED PERFORMANCE !!")
            self.best_perf["fitness"] = np.amax(self.fitness_arr).item()
            self.best_perf["eval_metrics"] = copy.deepcopy(
                self.metrics_arr[np.argmax(self.fitness_arr)]
            )
            self.best_perf["ansatz"] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)]
            )
            self.best_perf["ansatz_dicts"] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)].ansatz_dicts
            )
            self.best_perf["ansatz_draw"] = copy.deepcopy(
                self.population[np.argmax(self.fitness_arr)].ansatz_draw
            )
            self.best_perf["generation"] = gen
            self.best_perf["index"] = np.argmax(self.fitness_arr).item()

    def make_results(self, gen):
        start_time = time.time()
        ### Euclidean distances ###
        distances_from_best = euclidean_distances(self.population[np.argmax(self.fitness_arr)], self.population)
        destdir_curves = os.path.join(self.ga_output_path, "ga_curves", "run-%s" % self.start_time)
        if not os.path.exists(destdir_curves):
            os.makedirs(destdir_curves)
        filepath_euclid = os.path.join(
            destdir_curves,
            "%03deuclid_distance_data.png"
            % gen
        )
        plt.figure(0)
        plt.style.use("seaborn")
        plt.scatter(distances_from_best, self.fitness_arr, marker=".", c=[i["auroc"] for i in self.metrics_arr], cmap=plt.set_cmap('plasma'))
        cbar = plt.colorbar()
        cbar.set_label("AUROC")
        plt.ylabel("AUROC")
        plt.xlabel("Euclidian distance")
        plt.title("Euclidean Distances from Best Performing Ansatz")
        plt.savefig(filepath_euclid, format="png")
        plt.close(0)
        ### tSNE clustering ###
        data_tsne = tsne(self.population, rng_seed=self.rng_seed, perplexity=2)
        x, y = data_tsne[0], data_tsne[1]
        filepath_tsne = os.path.join(
            destdir_curves,
            "%03dtsne_distance_data.png"
            % gen
        )
        plt.figure(1)
        plt.style.use("seaborn")
        plt.scatter(x, y, marker=".", c=[i["auroc"] for i in self.metrics_arr], cmap=plt.set_cmap('plasma'))
        plt.ylabel("a.u.")
        plt.xlabel("a.u.")
        cbar = plt.colorbar()
        cbar.set_label("AUROC")
        plt.title("tSNE of Current Population")
        plt.savefig(filepath_tsne, format="png")
        plt.close(1)

        results = {
            "full_population": [i.ansatz_dicts for i in self.population],
            "full_drawn_population": [i.ansatz_draw for i in self.population],
            "full_fitness": [i for i in self.fitness_arr],
            "fitness_stats": f"Avg fitness: {np.mean(self.fitness_arr)}, Std. Dev: {np.std(self.fitness_arr)}",
            "full_eval_metrics": self.metrics_arr,
            "eval_metrics_stats": [
                f"Avg {k}: {np.mean([i[k] for i in self.metrics_arr])}, "
                + f"Std. Dev: {np.std([i[k] for i in self.metrics_arr])}"
                for k in self.metrics_arr[0].keys()
            ],
            # "full_distances": distances_from_best,
            # "distances_stats": f"Avg distance: {np.mean(distances_from_best)}, Std. Dev: {np.std(distances_from_best)}",
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
        for _ in range(self.n_winners):
            winner_ix = np.argmax(self.fitness_arr)
            winner = self.population[winner_ix]
            winner_arr.append(winner)
            self.fitness_arr.pop(winner_ix)

        self.population = []
        return winner_arr

    def mate(self, parents):
        """
        Swaps the qubits of ansatz.

        TODO: Implement gate mating.
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
            children = {
                "child_A": copy.deepcopy(parents[swap_ix[0]]), 
                "child_B": copy.deepcopy(parents[swap_ix[1]])
            }

            print(f"Pre-mateswap ansatz: {children}")

            moment_A = self.rng.integers(children["child_A"].n_moments) 
            moment_B = self.rng.integers(children["child_B"].n_moments)
            qubit_A = qubit_B = self.rng.integers(self.n_qubits).item()
            swap_set.add(qubit_A)

            def check_gate(qubit_str, qubit_num):
                if qubit_str.find("_") > 0:
                    return int(qubit_str[-1])
                return -1

            while True:
                A_link = check_gate(
                    children["child_A"][moment_A, qubit_A],
                    qubit_A
                )
                B_link = check_gate(
                    children["child_B"][moment_B, qubit_B],
                    qubit_B
                )

                if (A_link == qubit_A and B_link == qubit_B) or (A_link in swap_set and B_link in swap_set):
                    break
                else:
                    if A_link >= 0:
                        swap_set.add(A_link)
                        qubit_B = A_link
                    if B_link >= 0:
                        swap_set.add(B_link)
                        qubit_A = B_link

            for qubit in swap_set:
                children["child_A"][moment_A, qubit] = parents[swap_ix[1]][moment_B, qubit]
                children["child_B"][moment_B, qubit] = parents[swap_ix[0]][moment_A, qubit]

            print(f"Post-mateswap ansatz: {children}")

            children_arr.extend(children)

        for child in children_arr:
            self.mutate(child)

        return children_arr

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
        ansatz_backup = copy.deepcopy(ansatz)
        n_mutations = self.n_mutations
        count = 0
        while True:
            if count == 5:
                n_mutations += 1
                count = 0
            for _ in range(n_mutations):
                if (
                    ansatz.n_moments < self.max_moments
                    and self.rng.random() < self.add_moment_prob
                ):
                    ansatz.add_moment(method='duplicate')
                
                moment = self.rng.integers(ansatz.n_moments)
                qubit = self.rng.integers(self.n_qubits)
                ansatz.mutate(moment, qubit)

            ansatz_vector = tuple(create_vector(ansatz, return_type='list'))
            if ansatz_vector in self.set_of_all_circuits:
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
            ansatz_vectors_arr = []
            distances_arr = []
            for __ in range(100):
                while True:
                    ansatz_vector = tuple(create_vector(ansatz, return_type='list'))
                    if ansatz_vector in self.set_of_all_circuits:
                        ansatz = Individual(
                            self.n_qubits,
                            # self.rng.integers(1, self.max_moments + 1),
                            self.max_moments,
                            self.genepool,
                            self.rng_seed,
                        )
                        continue

                    ansatz_vectors_arr.append(ansatz_vector)
                    break
            for j in range(100):
                distances_arr.append(
                    np.array(
                        euclidean_distances(
                            ansatz_vectors_arr[j], 
                            self.set_of_all_circuits, 
                            max_moments=self.max_moments
                        )
                    )
                )
            ansatz = ansatz_vectors_arr[np.argmax(np.mean(distances_arr, axis=0))]
            immigrant_arr.append(
                ansatz
            )

        return immigrant_arr
            

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
