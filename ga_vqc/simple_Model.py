import copy
import datetime
import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from .Distance import euclidean_distances, string_distances, tsne
from .GA_ABC import GA_Model
from .GA_Support import make_results_json
from .simple_Individual import Individual
from .simple_Config import Config


class Model(GA_Model):
    """
    Container for all the logic of the GA Model.

    TODO: change the I assignment to a random assignment.
    """

    def __init__(self, config,  OUTPUT=True, DEBUG=False):
        """
        Params:
            - config: the configuration object for the model, details in simple_Config.py
            - OUTPUT: boolean that toggles whether model should print out anything (like progress
                updates) during training
        """
        self.DEBUG = DEBUG
        self.OUTPUT = OUTPUT

        self.config = config

        self.best_perf_arr = []

        self.population = []
        self.temp_pop = []
        self.fitness_arr = []
        self.metrics_arr = []

        self.set_of_all_circuit_diagrams = set(self.config.dict_of_preran_circuits.keys())
        self.set_of_all_circuit_vectors = set() # Add in preran circuit vectors at beginning

        self.start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Following time variables for debugging purposes only #
        self.total_ga_time = 0
        self.total_vqc_time = 0
        self.retrain_time = 0

        self.rng = np.random.default_rng(seed=self.config.rng_seed)

        ### hyperparams for qae ###
        self.vqc_config = config.vqc_config
        self.vqc_config["start_time"] = self.start_time

        self.full_population = [self.generate_ansatz(diagram=ansatz) 
                                for ansatz in self.config.dict_of_preran_circuits.keys()]
        self.set_of_all_circuit_vectors = set([tuple(ansatz.vector) 
                                               for ansatz in self.full_population])
        self.full_fitness_arr = [ansatz["fitness_metrics"] 
                                 for ansatz in self.config.dict_of_preran_circuits.values()]
        self.full_metrics_arr = [ansatz["eval_metrics"] 
                                 for ansatz in self.config.dict_of_preran_circuits.values()]
        self.per_gen_diversity = []
        self.full_pop_diversity = []
        
        self.generate_initial_pop()
    
    def new_best_perf(self, fitness_metrics=None, eval_metrics=None, 
                      ansatz=None, dicts=[], diagram=str(), generation=0, index=0):
        if fitness_metrics is None:
            fitness_metrics = self.config.dict_of_preran_circuits[self.config.empty_circuit_diagram]["fitness_metrics"]
        if eval_metrics is None:
            eval_metrics = self.config.dict_of_preran_circuits[self.config.empty_circuit_diagram]["eval_metrics"]
        self.best_perf_arr.append({
            "fitness_metrics": fitness_metrics,
            "eval_metrics": eval_metrics,
            "ansatz": ansatz,
            "dicts": dicts,
            "diagram": diagram,
            "generation": generation,
            "index": index,
        })
        
    
    def update_best_perf(self, best_keep_ixs):
        """
        Manages the upkeep of the self.best_perf array which tracks the best circuits the GA has found.

        Params:
            - best_keep_ixs: the indeces of the current self.best_perf_arr thast will remain in the updated
                self.best_perf_arr
        """
        temp_best_perf_arr = []
        for i in range(len(self.best_perf_arr)):
            if i in best_keep_ixs:
                temp_best_perf_arr.append(copy.deepcopy(
                    self.best_perf_arr[i]
                ))

        unsorted_ix_set = set([i for i in range(len(temp_best_perf_arr))])
        sorted_ixs = []
        while len(sorted_ixs) < len(temp_best_perf_arr):
            unsorted_ix_list = list(unsorted_ix_set)
            largest_ix = unsorted_ix_list[0]

            for i in unsorted_ix_list:
                if self.config.compare_fitness(temp_best_perf_arr[i]["fitness_metrics"], 
                        temp_best_perf_arr[largest_ix]["fitness_metrics"]):
                    largest_ix = i
            
            sorted_ixs.append(largest_ix)
            unsorted_ix_set.remove(largest_ix)

        self.best_perf_arr = [copy.deepcopy(temp_best_perf_arr[i]) for i in sorted_ixs]

    def generate_ansatz(self, dicts=None, diagram=None):
        """
        Handles the generation/construction of new ansatz - serves no purpose other than 
        cleaning up the code by moving the large constructor calls into 1 central function.

        Params:
            - dicts: optional parameter that constructs a new ansatz from a list of dictionaries 
                of moments (more info in simple_Individual.py)
            - diagram: optional parameter that constructs a new ansatz from a circuit diagram (more 
                info in simple_Individual.py)
        """

        if dicts is None and diagram is None:
            ansatz = Individual(
                        self.config.n_qubits,
                        self.config.max_moments,
                        self.config.max_vector_moments,
                        self.config.genepool,
                        self.config.rng_seed,
                    )
        elif dicts is not None:
            ansatz = Individual(
                        self.config.n_qubits,
                        self.config.max_moments,
                        self.config.max_vector_moments,
                        self.config.genepool,
                        self.config.rng_seed,
                        dicts=dicts
                    )
        elif diagram is not None:
            ansatz = Individual(
                        self.config.n_qubits,
                        self.config.max_moments,
                        self.config.max_vector_moments,
                        self.config.genepool,
                        self.config.rng_seed,
                        diagram=diagram
                    )
        else:
            raise ValueError('You must choose to generate an ansatz through' + 
                             'EITHER dicts OR diagram. Not both.')
        if ansatz.n_moments > self.config.max_vector_moments:
            self.update_circuit_set(2 * ansatz.n_moments)
        return ansatz
    
    def update_circuit_set(self, max_vector_moments):
        """
        Updates the vector set of all optimized circuits, so that the GA doesn't re-run old circuits
        AND so that new random circuits can be maximally different from previously run ones.
            -> !!! Not currently used in the GA to track anything: tracking handled with diagram set !!!

        Params:
            - max_vector_moments: the highest number of moments in all of the run circuits, important 
                for vector generation.
        """

        self.config.max_vector_moments = max_vector_moments
        self.set_of_all_circuit_vectors = set()
        for ansatz in self.full_population+self.temp_pop:
            ansatz.generate_vector(max_vector_moments)
            self.set_of_all_circuit_vectors.add(tuple(ansatz.vector))

    def clean(self):
        """
        Resets the self.temp_pop variable. Serves no purpose than to clean up code by renaming
        the resetting into a more understandable name.
        """

        self.temp_pop = []

    def distance(self, circuit_A, population, INIT=False):
        """
        Calculates the distance between a test circuit and a population of other circuits. For more info
        on how distances are calculated, see Distance.py

        Params:
            - circuit_A: test circuit to calculate distance with respect to
            - population: the population of circuits to compare circuit_A to
            - INIT: boolean that toggles whether the distance calculation is from the initial 
                population generation. Useful because the number of initial circuits makes it advantageous to
                use a quick-and-dirty comparison method as compared to calculating distances during training.
        """

        if (INIT and self.config.init_distance_method == 'euclidean') or (not INIT and self.config.distance_method == 'euclidean'):
            return np.mean(
                    euclidean_distances(
                        circuit_A,
                        self.generate_ansatz(diagram=self.config.empty_circuit_diagram),
                        population,
                    )
                ).item()
        elif (INIT and self.config.init_distance_method == 'string') or (not INIT and self.config.distance_method == 'string'):
            return np.mean(
                    string_distances(
                        circuit_A,
                        self.config.empty_circuit_diagram,
                        population,
                    )
                ).item()
        else:
            raise ValueError('distance_method must be either \'euclidean\' or \'string\'.')
        
    def generate_initial_pop(self):
        """
        Generates the initial population by making many more individuals than needed, and 
        pruning down to pop_size by taking maximally different individuals.
        """

        start_time = time.time()
        self.new_best_perf()
        for _ in range(self.config.init_pop_size):
            self.temp_pop.append(
                self.generate_ansatz()
            )

        ## Euclidean Distance on Vectorized Circuits ##
        selected_ixs = set()
        for i in range(self.config.pop_size):
            if i == 0:
                random_ix = self.rng.integers(0, self.config.init_pop_size)
                self.population.append(self.temp_pop[random_ix])
                continue

            distance_arr = []
            for j in range(len(self.population)):
                distance_arr.append(
                    self.distance(
                        self.population[j],
                        self.temp_pop,
                        INIT=True
                    )
                )

            sorted_ixs = np.argsort(distance_arr)
            selected_ix = sorted_ixs[-1]
            k = 1
            while selected_ix in selected_ixs:
                k += 1
                selected_ix = sorted_ixs[-k]
            selected_ixs.add(selected_ix)
            self.population.append(self.temp_pop[selected_ix])
        
        self.clean()
        end_time = time.time()
        exec_time = end_time - start_time
        self.total_ga_time += exec_time
        if self.OUTPUT:
            print(f"Initial generation/selection in {exec_time:.2f} seconds")

    def evolve(self):
        """
        Evolves the GA.
        """

        gen = 0
        while True:
            if self.OUTPUT:
                print(f"GA iteration {gen}")

            # Evaluating Fitness (Running VQCs) # 
            vqc_start_time = time.time()
            self.evaluate_fitness(gen) if not self.DEBUG else self.debug_eval_fitness(gen)
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
            if self.OUTPUT:
                i = 0
                for best_perf in self.best_perf_arr:
                    print(
                        f"{'-'*30}"
                        f"index: {i}, \n" +
                        f"Best fitness metrics: {[f'{key}: {val:.03f}' for key, val in best_perf['fitness_metrics'].items()]}, \n" + 
                        f"Best eval metrics: {[f'{key}: {val:.03f}' for key, val in best_perf['eval_metrics'].items()]}, \n" +
                        f"Best ansatz: \n{best_perf['diagram']}"
                    )
                    i+=1

            break_flag = False
            for best_perf in self.best_perf_arr:
                if (gen - best_perf["generation"]) > self.config.n_steps_patience:
                    break_flag = True
                    break
            if break_flag:
                break
            if self.OUTPUT:
                print(
                    "filepath is: ",
                    make_results_json(results, self.start_time, self.config.ga_output_path, gen)
                )
            gen += 1

            post_process_end_time = time.time()
            self.total_ga_time += (post_process_end_time - post_process_start_time)
            running_total_time = self.total_ga_time + self.total_vqc_time
            if self.OUTPUT:
                print(f'GA (classical) computations have taken {self.total_ga_time:0.2f} seconds \n     and {100 * self.total_ga_time / running_total_time:0.2f}% of the total time so far.')
                print(f'VQC (quantum) computations have taken {self.total_vqc_time:0.2f} seconds \n     and {100 * self.total_vqc_time / running_total_time:0.2f}% of the total time so far.')
        if self.OUTPUT:
            print(
                "filepath is: ",
                make_results_json(
                    results, self.start_time, self.config.ga_output_path, gen, final_flag=True
                ),
            )

        # post_process_end_time = time.time()
        # self.total_ga_time += (post_process_end_time - post_process_start_time)

        # if self.vqc_config["n_retrains"] < 20:
        #     ### Final re-training for std. dev. estimate ###
        #     retrain_start_time = time.time()

        #     for best_perf in self.best_perf_arr:
        #         ansatz = best_perf["ansatz"]
        #         vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
        #         vqc_config_ansatz["n_retrains"] = 20
        #         vqc_config_ansatz["dicts"] = ansatz.dicts
        #         vqc_config_ansatz["qml"] = ansatz.qml
        #         vqc_config_ansatz["diagram"] = ansatz.diagram
        #         vqc_config_ansatz["params"] = ansatz.params
        #         vqc_config_ansatz["gen"] = gen
                
        #         final_output = self.config.vqc(vqc_config_ansatz)
        #         final_fitness = final_output["fitness_metrics"]
        #         final_metrics = final_output["eval_metrics"]
        #         if self.OUTPUT:
        #             print(f"Final fitness: {final_fitness}")
        #             for metric in final_metrics.keys():
        #                 print(f"Final {metric}: {final_metrics[metric]}")

        #     retrain_end_time = time.time()
        #     self.retrain_time = retrain_end_time - retrain_start_time

        TOTAL_TIME = self.total_ga_time + self.total_vqc_time + self.retrain_time
        if self.OUTPUT:
            print(f'Final GA (classical) time: {self.total_ga_time:0.2f} seconds')
            print(f'GA fraction of total time: {100 * self.total_ga_time / TOTAL_TIME:0.2f} %')
            print(f'Final VQC (quantum) time: {self.total_vqc_time:0.2f} seconds')
            print(f'VQC fraction of total time: {100 * self.total_vqc_time / TOTAL_TIME:0.2f} %')
            # print(f'Final retrain (quantum, for final statistics) time: {self.retrain_time:0.2f} seconds')
            # print(f'Retrain fraction of total time: {100 * self.retrain_time / TOTAL_TIME:0.2f} %')

    def evaluate_fitness(self, gen):
        """
        Evaluates the fitness level of all ansatz. Runs the QML optimization task.

        Params:
            - gen: the current generation of the GA
        """

        # Setup function args for multithreading #
        ix = 0
        args_arr = []
        for ansatz in self.population:
            self.full_population.append(ansatz)
            self.set_of_all_circuit_vectors.add(tuple(ansatz.vector))
            self.set_of_all_circuit_diagrams.add(ansatz.diagram)

            vqc_config_ansatz = {key: value for key, value in self.vqc_config.items()}
            vqc_config_ansatz["dicts"] = ansatz.dicts
            vqc_config_ansatz["qml"] = ansatz.qml
            vqc_config_ansatz["diagram"] = ansatz.diagram
            vqc_config_ansatz["params"] = ansatz.params
            vqc_config_ansatz["ix"] = ix
            vqc_config_ansatz["gen"] = gen
            args_arr.append(copy.deepcopy(vqc_config_ansatz))
            ix += 1
        
        # Diversity Tracking #
        gen_distance_arr = []
        for ansatz in self.population:
            gen_distance_arr.append(
                self.distance(
                    ansatz,
                    self.population
                )
            )
        self.per_gen_diversity.append(np.mean(gen_distance_arr))
        full_distance_arr = []
        for ansatz in self.full_population:
            full_distance_arr.append(
                self.distance(
                    ansatz,
                    self.full_population
                )
            )
        self.full_pop_diversity.append(full_distance_arr)

        # Optimize the current popualtion ansatz using multithreading #
        start_time = time.time()
        output_arr = []
        for i in range(self.config.pop_size // self.config.max_concurrent):
            with mp.get_context("spawn").Pool(processes=self.config.max_concurrent) as pool:
                output_arr.extend(
                    pool.map(
                        self.config.vqc,
                        args_arr[
                            i * self.config.max_concurrent : (i + 1) * self.config.max_concurrent
                        ],
                    )
                )
        
        # Both the fitness metrics and eval_metrics must be JSON serializable 
        #   -> (ie. default python classes or have custom serialization)
        # for _ in range(self.config.n_fitness_metrics):
        self.fitness_arr = [output["fitness_metrics"] for output in output_arr]
        self.full_fitness_arr.extend(self.fitness_arr)

        # for _ in range(self.config.n_eval_metrics):
        self.metrics_arr = [output["eval_metrics"] for output in output_arr]
        self.full_metrics_arr.extend(self.metrics_arr)

        end_time = time.time()
        exec_time = end_time - start_time
        self.total_vqc_time += exec_time
        if self.OUTPUT:
            print(f"QML Optimization in {exec_time:.2f} seconds")

        best_keep_ix_set, new_keep_ix_set = self.config.choose_best([best_perf['fitness_metrics'] for best_perf in self.best_perf_arr], self.fitness_arr)
        for ix in list(new_keep_ix_set):
            if self.OUTPUT:
                print('NEW GOOD CIRCUIT')
            self.new_best_perf(fitness_metrics=self.fitness_arr[ix], 
                               eval_metrics=copy.deepcopy(
                                    self.metrics_arr[ix]
                                ), 
                               ansatz=copy.deepcopy(
                                    self.population[ix]
                                ), 
                               dicts=copy.deepcopy(
                                    self.population[ix].dicts
                                ), 
                               diagram=copy.deepcopy(
                                    self.population[ix].diagram
                                ), 
                               generation=gen, 
                               index=ix)
            best_keep_ix_set.add(len(self.best_perf_arr)-1)
        # print(f'pre update best_perf_arr: {len(self.best_perf_arr)}')
        self.update_best_perf(best_keep_ix_set)
        # print(f'POST update best_perf_arr: {len(self.best_perf_arr)}')

    def debug_eval_fitness(self, gen):
        self.fitness_arr = []
        self.metrics_arr = []
        for _ in range(len(self.population)):
            self.fitness_arr.append(
                {"avg_fitness": self.rng.random(),
                "stddev_fitness": self.rng.random()})
            self.metrics_arr.append(
                {"auroc": self.rng.random()}
            )

        best_keep_ix_set, new_keep_ix_set = self.config.choose_best([best_perf['fitness_metrics'] for best_perf in self.best_perf_arr], self.fitness_arr)
        for ix in list(new_keep_ix_set):
            if self.OUTPUT:
                print('NEW GOOD CIRCUIT')
            self.new_best_perf(fitness_metrics=self.fitness_arr[ix], 
                               eval_metrics=copy.deepcopy(
                                    self.metrics_arr[ix]
                                ), 
                               ansatz=copy.deepcopy(
                                    self.population[ix]
                                ), 
                               dicts=copy.deepcopy(
                                    self.population[ix].dicts
                                ), 
                               diagram=copy.deepcopy(
                                    self.population[ix].diagram
                                ), 
                               generation=gen, 
                               index=ix)
            best_keep_ix_set.add(len(self.best_perf_arr)-1)
        # print(f'pre update best_perf_arr: {len(self.best_perf_arr)}')
        self.update_best_perf(best_keep_ix_set)
        # print(f'POST update best_perf_arr: {len(self.best_perf_arr)}')

        
    def make_results(self, gen):
        """
        Generates the output data (graphs + dictionary) and files.

        Params:
            - gen: the current generation of the GA 
        """

        ### tSNE clustering ###
        destdir_curves = os.path.join(self.config.ga_output_path, "ga_curves", "run-%s" % self.start_time)
        if not os.path.exists(destdir_curves):
            os.makedirs(destdir_curves)
        data_tsne_arr = []
        for perp in range(2, len(self.full_population)):
            data_tsne = tsne(self.full_population, rng_seed=self.config.rng_seed, perplexity=perp)
            data_tsne_arr.append(data_tsne.tolist())
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
            "vqc_optimization_time": self.total_vqc_time,
            "ga_post-processing_time": self.total_ga_time,
            "vqc_retrain_time": self.retrain_time,
            "full_population_vectors": [ansatz.vector for ansatz in self.full_population],
            "full_population_drawings": [ansatz.diagram for ansatz in self.full_population],
            "final_max_vector_moments": self.config.max_vector_moments,
            "full_population_fitness": self.full_fitness_arr,
            "full_population_metrics": self.full_metrics_arr,
            "full_tsne_data": data_tsne_arr,
            "per_gen_diversity": self.per_gen_diversity,
            "full_pop_diversity": self.full_pop_diversity,
            "current_generation": [i.dicts for i in self.population],
            "current_drawn_generation": [i.diagram for i in self.population],
            "current_generation_fitness": [i for i in self.fitness_arr],
            "current_generation_eval_metrics": [i for i in self.metrics_arr],
            "best_ansatzes": [best_perf["dicts"] for best_perf in self.best_perf_arr],
            "best_drawn_ansatzes": [best_perf["diagram"] for best_perf in self.best_perf_arr],
            "best_fitnesses": [best_perf["fitness_metrics"] for best_perf in self.best_perf_arr],
            "best_fitness_gens": [best_perf["generation"] for best_perf in self.best_perf_arr],
            "best_fitness_ixs": [best_perf["index"] for best_perf in self.best_perf_arr],
            "best_eval_metrics": [best_perf["eval_metrics"] for best_perf in self.best_perf_arr]
        }   

        return results

    def select(self):
        """
        Picks the top performing ansatz from a generation to mate and mutate for the next generation.
        """
        winner_arr = []
        for _ in range(self.config.n_winners):
            # winner_ix = np.argmax(self.fitness_arr)
            winner_ix = 0
            for i in range(len(self.fitness_arr)):
                if self.config.compare_fitness(self.fitness_arr[i], self.fitness_arr[winner_ix]):
                    winner_ix = i
            winner = self.population[winner_ix]
            winner_arr.append(winner)
            self.fitness_arr.pop(winner_ix)

        self.population = []
        return winner_arr

    def mate(self, parents, num_children):
        """
        Swaps the qubits of ansatz.

        Params:
            - parents: array of circuits from old generation that will mate to create 
                the new generation
            - num_children: number of children for each pair of parents
        """

        children_arr = []
        swap_ixs = []
        parents = self.deep_permutation(parents)
        while len(parents) < self.config.pop_size - self.config.n_new_individuals:
            parents.extend(self.deep_permutation(parents))

        # Create index pairings for swapping
        for j in range(len(parents) // self.config.n_winners):
            for i in range(self.config.n_winners):
                if i % 2 != 0:
                    continue
                swap_ixs.append(
                    {
                        "parent_A": (j * self.config.n_winners) + i, 
                        "parent_B": (j * self.config.n_winners) + i + 1
                    }
                )

        # Perform the swap with neighboring parents
        swap_set = set()
        for swap_ix in swap_ixs:  # NOT setup for odd number of parents
            child_A = copy.deepcopy(parents[swap_ix["parent_A"]])
            child_B = copy.deepcopy(parents[swap_ix["parent_B"]])
            self.temp_pop.append(child_A)
            self.temp_pop.append(child_B)

            moment_A = self.rng.integers(child_A.n_moments) 
            moment_B = self.rng.integers(child_B.n_moments)
            qubit_A = qubit_B = self.rng.integers(self.config.n_qubits).item()
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
            if child_A.n_vector_moments > self.config.max_vector_moments or child_B.n_vector_moments > self.config.max_vector_moments:
                self.update_circuit_set(max(child_A.n_vector_moments, child_B.n_vector_moments))
            children_arr.extend([child_A, child_B])

        for child in children_arr:
            self.mutate(child)
        self.clean()

        return children_arr[:num_children]

    def deep_permutation(self, arr):
        """
        Piggybacks on numpy.permutation to perform permutations on arrays of custom user-defined types
        without having to duplicate in memory - ie. just permutes at the array level with the memory addresses
        of the underlying objects.
        """

        arr_copy = [i for i in arr]
        ix_arr = self.rng.permutation(len(arr))
        for i in range(len(arr)):
            arr[i] = arr_copy[ix_arr[i]]

        return arr

    def mutate(self, ansatz, n_mutations=None):
        """
        Mutates a single ansatz by modifying n_mutations random qubit(s) at 1 random time each.

        Params:
            - ansatz: the circuit to mutate
            - n_mutations: optional parameter for the number of mutations to perform on 
                the circuit

        Variables
            j: selected moment
            i: selected qubit
            k: selected gate
        """
        ansatz_backup = copy.deepcopy(ansatz)
        if n_mutations is None:
            n_mutations = self.config.n_mutations
        count = 0
        while True:
            if count == 5:
                n_mutations += 1
                count = 0
            for _ in range(n_mutations):
                moment = self.rng.integers(ansatz.n_moments)
                qubit = self.rng.integers(self.config.n_qubits)
                ansatz.mutate(moment, qubit)

            if ansatz.n_vector_moments > self.config.max_vector_moments:
                self.update_circuit_set(ansatz.n_vector_moments)
            if ansatz.diagram in self.set_of_all_circuit_diagrams:
                ansatz = copy.deepcopy(ansatz_backup)
                count += 1
                continue
            break

    def immigrate(self, n_individuals):
        """
        Adds in new individuals with every generation, in order to keep up the overall 
        population diversity. Overgenerates circuits and selects the ones with maximum
        distance to previously run circuits.

        Params:
            - n_individuals: number of new circuits to generate
        """
        immigrant_arr = []
        for _ in range(n_individuals):
            ansatz_arr = []
            distance_arr = []
            for __ in range(100):
                ansatz = self.generate_ansatz()
                count = 0
                while True:
                    if count == 10:
                        self.config.max_moments += 1
                        ansatz = self.generate_ansatz()
                        count = 0
                    self.temp_pop.append(ansatz)
                    if ansatz.diagram in self.set_of_all_circuit_diagrams:
                        ansatz = self.generate_ansatz()
                        count += 1
                        continue

                    ansatz_arr.append(ansatz)
                    break
            for j in range(100):
                distance_arr.append(
                    self.distance(
                        ansatz_arr[j],
                        self.full_population
                    )
                )
            ansatz = ansatz_arr[np.argmax(distance_arr)]
            immigrant_arr.append(
                ansatz
            )
            self.clean()

        return immigrant_arr
            
    def anneal_manager(self, gen):
        """
        Handles the 'annealing' portion of the GA. In principle this would change the fraction of 
        children and immigrants created as the GA evolves.
            -> !!! Not currently used to do real annealing the GA: kept as an example 
            of what could be done !!!
        """
        
        num_children = self.config.pop_size // 2
        num_immigrants = self.config.pop_size - num_children
        
        return num_children, num_immigrants
