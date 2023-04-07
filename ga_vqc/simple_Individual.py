import copy

import numpy as np
import pennylane as qml

from .GA_ABC import GA_Individual


class Individual(GA_Individual):
    """
    Data structure for the ansatz.
    """

    def __init__(self, n_qubits, n_moments, genepool, rng_seed):
        """
        Initializes the inidivual object, all individuals (ansatz) are members of this class.
            - n_qubits: Number of qubits in the ansatz.
            - n_moments: Number of moments in the ansatz.
            - genepool: All the allowed gates for the ansatz, as well as meta-data 
                            about those gates.

        TODO: change params to allow for multi-param gates
        """
        self.n_qubits = n_qubits
        self.n_moments = n_moments
        self.genepool = genepool
        self.rng = np.random.default_rng(seed=rng_seed)

        self.ansatz_dicts = []
        self.ansatz_qml = []
        self.ansatz_draw = []
        self.n_params = 0

        self.generate()  # Should this be done automatically?
        self.convert_to_qml()
        self.draw_ansatz()

    def __len__(self):
        return len(self.ansatz_dicts[0])

    def __getitem__(self, key):
        if type(key) is tuple:
            return self.ansatz_dicts[int(key[0])][int(key[1])]
        else:
            return self.ansatz_dicts[int(key)]

    def __setitem__(self, key, value):
        """
        TODO: fix implementation (need to set whole moments?)
        """
        if type(key) is tuple:
            self.ansatz_dicts[int(key[0])][int(key[1])] = value
        else:
            self.ansatz_dicts[int(key)] = value

    def __str__(self):
        return str(self.ansatz_dicts)

    def __repr__(self):
        return self.__str__()

    def generate(self):
        """
        Generates the ansatz stochastically based on the other member variables.
        """
        for moment in range(self.n_moments):
            self.ansatz_dicts.append(dict.fromkeys(range(self.n_qubits), 0))
            qubits = self.rng.permutation(self.n_qubits)  # which qubit to pick first
            for qubit in qubits:
                if self.ansatz_dicts[moment][qubit] != 0:
                    continue
                gate = self.genepool.choice()

                if gate.n_qubits == 1:
                    self.ansatz_dicts[moment][qubit] = gate.name
                    self.n_params += gate.n_params
                    continue
                elif gate.n_qubits == 2:
                    qubit_pairs = self.rng.permutation(
                        self.n_qubits
                    )
                    for qubit_pair in qubit_pairs:
                        if self.ansatz_dicts[moment][qubit_pair] != 0 or qubit_pair == qubit:
                            continue

                        direction = self.rng.permutation(["_C", "_T"])
                        self.ansatz_dicts[moment][qubit] = gate.name + direction[0] + f"-{qubit_pair}"
                        self.ansatz_dicts[moment][qubit_pair] = gate.name + direction[1] + f"-{qubit}"
                        self.n_params += gate.n_params
                        break

                    if self.ansatz_dicts[moment][qubit] == 0:
                        gate = self.genepool.choice(n_qubits=1)
                        self.ansatz_dicts[moment][qubit] = gate.name
                        self.n_params += gate.n_params
                else:
                    raise Exception("Gates with more than 2 qubits haven't been implemented yet.")

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
        for moment in range(len(self.ansatz_dicts)):
            moment_dict = {gate.name: list() for gate in self.genepool.gates}
            stored_i = []
            for qubit in range(self.n_qubits):
                _ix = self.ansatz_dicts[moment][qubit].find("_")
                if _ix < 0:
                    moment_dict[self.ansatz_dicts[moment][qubit]].append(qubit)
                else:
                    if qubit in stored_i:
                        continue
                    _1ix = self.ansatz_dicts[moment][qubit][_ix + 1]
                    q_p = int(self.ansatz_dicts[moment][qubit][-1])
                    stored_i.append(q_p)
                    if _1ix == "C":
                        moment_dict[self.ansatz_dicts[moment][qubit][:_ix]].append([qubit, int(q_p)])
                    else:
                        moment_dict[self.ansatz_dicts[moment][qubit][:_ix]].append([int(q_p), qubit])

            count = 0
            for gate_name in moment_dict.keys():
                if len(moment_dict[gate_name]) == 0 or gate_name == "I":
                    continue
                if self.genepool.n_qubits(gate_name) == 1: # Change to check n_params of gate
                    self.ansatz_qml.append(
                        f"qml.broadcast(qml.{gate_name}, wires={moment_dict[gate_name]}, pattern='single', " +
                        f"parameters=params[{count}:{count + (self.genepool.n_params(gate_name) * len(moment_dict[gate_name]))}])"
                    )
                    count += self.genepool.n_params(gate_name) * len(moment_dict[gate_name])
                elif self.genepool.n_qubits(gate_name) == 2:  # Assumes the 2-qubit gates have no parameters, which is not generally true
                    self.ansatz_qml.append(
                        f"qml.broadcast(qml.{gate_name}, wires={np.array(moment_dict[gate_name]).flatten(order='C').tolist()}, " +
                        f"pattern={moment_dict[gate_name]})"
                    )
                    # change to allow for two-qubit gates with 1+ params

    def ansatz_circuit(self, params, event=None):
        for m in self.ansatz_qml:
            exec(m)
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
        )([0.0 for _ in range(self.n_params)], event=[i for i in range(self.n_qubits)])[:-3]
        indices = [i for i, c in enumerate(full_ansatz_draw) if c == ":"]
        for _ in range(self.n_qubits):
            self.ansatz_draw.append(full_ansatz_draw[: indices[1] - 2][3:-2])
            full_ansatz_draw = full_ansatz_draw[indices[1] - 1 :]

    def add_moment(self, method='random'):
        """
        TODO: change to randomly generate new moment?
        """
        moment = self.rng.integers(self.n_moments)

        if method == 'random':
            pass
        elif method == 'duplicate':
            self.ansatz_dicts.append(copy.deepcopy(self.ansatz_dicts[moment]))
        elif method == 'pad':
            pass
        else:
            raise Exception("Method not supported.")
        
        self.n_moments += 1

    def mutate(self, moment, qubit):
        """
        Mutates a single ansatz by modifying n_mutations random qubit(s) at 1 random time each.

        TODO:

        Variables
            j: selected moment
            i: selected qubit
            k: selected gate
        """
        double_swap_flag = 0
        gate = self.genepool.choice()

        if self.ansatz_dicts[moment][qubit].find("_") > 0:
            double_swap_flag += 1
            gate_pair = self.genepool.choice(n_qubits=1)
            self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit][-1])] = gate_pair.name

        if gate.n_qubits == 1:
            self.ansatz_dicts[moment][qubit] = gate.name
        else:
            qubit_pairs = self.rng.permutation(self.n_qubits)
            for qubit_pair in qubit_pairs:
                if qubit_pair == qubit:
                    continue
                if self.ansatz_dicts[moment][qubit_pair].find("_") > 0:
                    double_swap_flag += 1
                    gate_pair = self.genepool.choice()
                    if double_swap_flag == 2 and gate_pair.n_qubits == 1:
                        direction = self.rng.permutation(["_C", "_T"])
                        self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit][-1])] = (
                            gate.name + direction[0] + f"-{int(self.ansatz_dicts[moment][qubit_pair][-1])}"
                        )
                        self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit_pair][-1])] = (
                            gate.name + direction[1] + f"-{int(self.ansatz_dicts[moment][qubit][-1])}"
                        )
                    else:
                        self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit_pair][-1])] = self.rng.choice(
                            self.gates_arr[:-1]
                        )

                direction = self.rng.permutation(["_C", "_T"])
                self.ansatz_dicts[moment][qubit] = gate.name + direction[0] + f"-{qubit_pair}"
                self.ansatz_dicts[moment][qubit_pair] = gate.name + direction[1] + f"-{qubit}"
                break