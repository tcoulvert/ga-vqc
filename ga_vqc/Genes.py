import numpy as np

class Gate:
    def __init__(self, name, n_qubits, n_params, constructor=None) -> None:
        # Must be a gate supported by Pennylane, 
        #  and name must be the pennylane name of said gate.
        self.name = name
        
        self.n_qubits = n_qubits
        self.n_params = n_params

        # Use to allow user to specify their own gate
        self.constructor = constructor

    def __str__(self) -> str:
        return self.name

    def __repr__(self):
        return self.__str__()

class Genepool:
    def __init__(self, gates, probs, rng_seed=None) -> None:
        self.gates = []
        self.probs = probs
        self.rng = np.random.default_rng(rng_seed)

        self.generate_gates(gates)
        
    def generate_gates(self, gates) -> None:
        for k, v in gates.items():
            self.gates.append(Gate(k, v[0], v[1]))

    def gate_list(self) -> list:
        return [gate.name for gate in self.gates]

    def n_params(self, gate_name) -> int:
        if gate_name is None:
            raise Exception("Cannot look for None-type gate.") 
        for gate in self.gates:
            if gate.name == gate_name:
                return gate.n_params

        raise Exception("Gate asked for not in genepool.")

    def n_qubits(self, gate_name) -> int:
        if gate_name is None:
            raise Exception("Cannot look for None-type gate.")
        if gate_name[0] == 'C': # properly check for non-name gates -> change ansatz_dicts to adjacency matrix
            return 2 
        for gate in self.gates:
            if gate.name == gate_name:
                return gate.n_qubits

        raise Exception("Gate asked for not in genepool.")

    def n_gates(self, search_param={}):
        """
        Returns total number of gates unless given another parameter to search for.
        """
        for k, v in search_param.items():
            count = 0
            if k == 'n_params':
                for gate in self.gates:
                    if gate.n_params == v:
                        count += 1
                return count
            elif k == 'n_qubits':
                for gate in self.gates:
                    if gate.n_qubits == v:
                        count += 1
                return count

        return len(self.gates)

    def index_of(self, gate_name):
        if gate_name is None:
            raise Exception("Cannot look for None-type gate.")
        for i in range(len(self.gates)):
            if self.gates[i].name == gate_name:
                return i

        raise Exception("Gate not in genepool.")

    def choice(self, size=1, replace=True, n_qubits=None, n_params=None):
        gates_copy = [gate for gate in self.gates]
        probs_copy = [prob for prob in self.probs]
        
        if n_qubits is not None:
            for i in range(len(gates_copy)):
                if gates_copy[i].n_qubits != n_qubits:
                    gates_copy.pop(i)
                    probs_copy.pop(i)

        if n_params is not None:
            for i in range(len(gates_copy)):
                if gates_copy[i].n_params != n_params:
                    gates_copy.pop(i)
                    probs_copy.pop(i)

        if len(gates_copy) == 0:
            raise Exception("No gate with the specified qualities.")

        probs_sum = np.sum(probs_copy)
        probs_copy = [prob / probs_sum for prob in probs_copy]
        
        if size == 1:
            return self.rng.choice(gates_copy, size=1, replace=replace, p=probs_copy).item()

        return self.rng.choice(gates_copy, size=size, replace=replace, p=probs_copy)

    def permute(self) -> np.ndarray:
        return self.rng.permutation(self.gates)

