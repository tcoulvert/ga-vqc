class Config():
    def __init__(self, vqc_main, vqc_config, genepool, ga_output_path, 
                 empty_circuit_data, dict_of_preran_circuits=None, rng_seed=None) -> None:
        """
            Defines the configuration parameters for the GA.

            GA Config Params:
                - backend_type: Type of backend to use for the GA, so far only high is implemented.
                                    In the future this will change how low-level the circuits are
                - vqc: Your function that runs the vqc training. Think like the main function for your vqc.
                - vqc_config: Dictionary that defines the configuration parameters for your vqc.
                - ga_output_path: filepath for the output of the GA.
                - max_concurrent: Used for parallelization, defines the number of concurrent circuits to 
                                    use multiprocessing on.
                - n_qubits: The number of qubits the GA optimize the ansatz with. This is static, and is 
                                determined by the dimensionality of your input data.
                - n_init_moments: The number of moments the GA will initialize for all circuits it tries.
                - max_moments: The number of maximum moments for all circuits. This calues changes as the GA
                                    populates more of the circuits. If a threshold # of circuits are at this 
                                    value (usually 80%), max_moments will increase. This is done to limit 
                                    bias placed on the GA.
                - add_moment_prob: The probability to add a moment to a circuit.
                - genepool: The Genepool, which encodes the allowed gates and their respective probabilities.
                - pop_size: Total number of circuits in the population at every optimization step.
                - init_pop_size: The number of circuits used for the initial string-distance randomization,
                                    which is done to maximize initial diversity.
                - n_new_individuals: Number of completely new circuits to generate every round. Can be 0.
                - n_winners: Number of population that are allowed to mate for the next round.
                - n_mutations: Number of point (individual gate) mutations to perform on the children 
                                circuits.
                - n_steps_patience: Number of optimization steps to wait with no improvement before ending
                                        the GA.
                - rng_seed: Seed for random operations, allows reproducibility.

        """
        self.backend_type = "high"
        self.vqc = vqc_main # main func that handles variational quantum circuit training
        self.vqc_config = vqc_config
        self.ga_output_path = ga_output_path
        
        self.max_concurrent = 1 # for multithreading/multiprocessing purposes

        self.n_qubits = 3
        self.n_init_moments = 2
        self.max_moments = 4 # >= 1
        self.max_vector_moments = self.n_qubits * self.max_moments
        self.add_moment_prob = 0.0

        self.genepool = genepool

        self.pop_size = 4 # must be a multiple of max_concurrent
        self.init_pop_size = 10
        self.n_new_individuals = 2  # >= 0
       
        self.n_winners = 2 # needs to be an even number
        self.n_mutations = 1
        self.n_steps_patience = 15
        self.n_eval_metrics = 0

        if dict_of_preran_circuits is None:
            self.dict_of_preran_circuits = dict()
        else:
            self.dict_of_preran_circuits = dict_of_preran_circuits

        self.empty_circuit_diagram = str()
        for q in range(self.n_qubits):
            if q != self.n_qubits - 1:
                self.empty_circuit_diagram += f'{q}: {"─"*8*self.max_moments}┤ \n'
                continue
            self.empty_circuit_diagram += f'{q}: {"─"*8*self.max_moments}┤  '
        self.dict_of_preran_circuits[
            self.empty_circuit_diagram
        ] = {
                "fitness_metric": empty_circuit_data["fitness_metric"],
                "eval_metrics": empty_circuit_data["eval_metrics"]
            }
        
        self.init_distance_method = 'euclidean'
        self.distance_method = 'string'
        
        self.rng_seed = rng_seed
        