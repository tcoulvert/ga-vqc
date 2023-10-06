import copy
import re

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from .GA_ABC import GA_Individual


class Individual(GA_Individual):
    """
    Data structure for the ansatz.
    """

    def __init__(self, n_qubits, n_moments, genepool, rng_seed, ansatz_dicts=None):
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
        self.pennylane_rng = pnp.random.default_rng(seed=rng_seed)

        self.ansatz_dicts = []
        self.ansatz_qml = []
        self.ansatz_draw = []
        self.params = []

        if ansatz_dicts is None:
            self.generate()
        else:
            self.generate_from(ansatz_dicts)

    def __len__(self):
        return len(self.ansatz_dicts[0])

    def __getitem__(self, key):
        if type(key) is tuple:
            return self.ansatz_dicts[int(key[0])][int(key[1])]
        else:
            return self.ansatz_dicts[int(key)]

    def __setitem__(self, key, value):
        """
        Allows editing of the ansatz via the [] operators.
        """
        if type(key) is tuple:
            self.ansatz_dicts[int(key[0])][int(key[1])] = value
        else:
            self.ansatz_dicts[int(key)] = value
        self.convert_to_qml()
        self.draw_ansatz()

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
                        break

                    if self.ansatz_dicts[moment][qubit] == 0:
                        gate = self.genepool.choice(n_qubits=1)
                        self.ansatz_dicts[moment][qubit] = gate.name
                else:
                    raise Exception("Gates with more than 2 qubits haven't been implemented yet.")
                
        self.convert_to_qml()
        self.draw_ansatz()

    def generate_from(self, ansatz):
        self.ansatz_dicts = copy.deepcopy(ansatz)
        self.convert_to_qml()
        self.draw_ansatz()

    def convert_to_qml(self):
        """
        Converts the ansatz into a general format for the QML.

            moment_dict example: _C (_T) means current qubit is control (target) qubit of 2-qubit gate
                {'I': [],
                 'RX': [],
                 'RY': [],
                 'RZ': [],
                 'CNOT': []}
        """
        self.ansatz_qml = []
        self.params = []
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

            for gate_name in moment_dict.keys():
                if len(moment_dict[gate_name]) == 0 or gate_name == "I":
                    continue
                if self.genepool.n_qubits(gate_name) == 1: # Change to check n_params of gate
                    if self.genepool.n_params(gate_name) > 0:
                        self.ansatz_qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={moment_dict[gate_name]}, pattern='single', " +
                            f"parameters=params[{len(self.params)}:{len(self.params) + len(moment_dict[gate_name])}])"
                        )
                        for _ in range(len(moment_dict[gate_name])):
                            # self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name), requires_grad=True))
                            self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name)))
                    else:
                        self.ansatz_qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={moment_dict[gate_name]}, pattern='single')"
                        )
                elif self.genepool.n_qubits(gate_name) == 2:
                    if self.genepool.n_params(gate_name) > 0:
                        self.ansatz_qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={np.array(moment_dict[gate_name]).flatten(order='C').tolist()}, " +
                            f"parameters=params[{len(self.params)}:{len(self.params) + len(moment_dict[gate_name])}])" +
                            f"pattern={moment_dict[gate_name]})"
                        )
                        for _ in range(len(moment_dict[gate_name])):
                            # self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name), requires_grad=True))
                            self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name)))
                    else:
                        self.ansatz_qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={np.array(moment_dict[gate_name]).flatten(order='C').tolist()}, " +
                            f"pattern={moment_dict[gate_name]})"
                        )
        self.params = pnp.array(self.params, dtype=object, requires_grad=True)

    def ansatz_circuit(self, params, event=None):
        for m in self.ansatz_qml:
            exec(m)
        return qml.expval(qml.PauliZ(wires=[self.n_qubits - 1]))

    def draw_ansatz(self):
        self.ansatz_draw = qml.draw(
            qml.QNode(
                qml.compile(self.ansatz_circuit),
                qml.device("default.qubit", wires=self.n_qubits, shots=1),
            ),
            decimals=None,
            expansion_strategy="device",
            show_all_wires=True,
        )(self.params, event=[i for i in range(self.n_qubits)])[:-3]

        self.update_ansatz_dicts()

    def update_ansatz_dicts(self):
        """
        Update the ansatz from the drawn ansatz after compiling the circuit. 

        TODO: Figure out how to make this general for any gate (or at 
                minimum any gate the compiler spits out)
        """
        self.ansatz_dicts = []

        ix_arr = [(0, -1)] + [m.span() for m in re.finditer('\n', self.ansatz_draw)]
        qubit_array = [self.ansatz_draw[ix_arr[i][1]+1 : ix_arr[i+1][0]] for i in range(len(ix_arr) - 1)]
        # print(qubit_array)

        q = 0
        for qubit in qubit_array:
            multi_qubit_gates = [m.start() for m in re.finditer('╭')] + [m.start() for m in re.finditer('╰')]
            multi_qubit_gates.sort()

            mask_ixs_set = set([m.start() for m in re.finditer('─')])
            all_ixs = list(range(len(qubit)))
            gate_ixs = [i for i in all_ixs if i not in mask_ixs_set]

            skip_flag = False
            moment = 0
            for j in gate_ixs:
                if skip_flag:
                    skip_flag = False
                    continue
                
                if len(self.ansatz_dicts) <= moment:
                    self.ansatz_dicts.append(dict.fromkeys(range(self.n_qubits), 0))

                # THIS LOGIC SPECIFIC TO ONE GENEPOOL (RX, RY, RZ, Rϕ, CNOT)
                if qubit[j] == 'R':
                    skip_flag = True

                    self.ansatz_dicts[moment][q] = 'R' + qubit[j+1]
                    moment += 1
                    continue

                if qubit[j] == '╭' or qubit[j] == '╰':
                    skip_flag = True

                    if qubit[j+1] == 'X':
                        self.ansatz_dicts[moment][q] = 'CNOT_T'
                    else:       # qubit[j+1] == '●'
                        self.ansatz_dicts[moment][q] = 'CNOT_C'
                    moment += 1
                    continue
            
            q += 1

        for moment_dict in self.ansatz_dicts:
            for k, v in moment_dict.items():
                if v == 0:
                    moment_dict[k] = 'I'

        self.n_moments = len(self.ansatz_dicts)
        self.convert_to_qml()

    def add_moment(self, method='random', **kwargs):
        """
        TODO: change to randomly generate new moment?
        """
        moment = self.rng.integers(self.n_moments)

        if method == 'random':
            raise Exception("Method not yet supported.")
        elif method == 'duplicate':
            self.ansatz_dicts.append(copy.deepcopy(self.ansatz_dicts[moment]))
            self.n_moments += 1
        elif method == 'pad':
            for _ in range(kwargs['num_pad']):
                self.ansatz_dicts.append(dict.fromkeys(range(self.n_qubits), 'I'))
                self.n_moments += 1
        else:
            raise Exception("Method not supported.")
        
        self.convert_to_qml()
        self.draw_ansatz()

    def mutate(self, moment, qubit):
        """
        Mutates a single ansatz by modifying n_mutations random qubit(s) at 1 random time each.

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
                    if double_swap_flag == 2 and gate_pair.n_qubits == 2:
                        direction = self.rng.permutation(["_C", "_T"])
                        self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit][-1])] = (
                            gate_pair.name + direction[0] + f"-{int(self.ansatz_dicts[moment][qubit_pair][-1])}"
                        )
                        self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit_pair][-1])] = (
                            gate_pair.name + direction[1] + f"-{int(self.ansatz_dicts[moment][qubit][-1])}"
                        )
                    else:
                        gate_pair = self.genepool.choice(n_qubits=1)
                        self.ansatz_dicts[moment][int(self.ansatz_dicts[moment][qubit_pair][-1])] = gate_pair.name

                direction = self.rng.permutation(["_C", "_T"])
                self.ansatz_dicts[moment][qubit] = gate.name + direction[0] + f"-{qubit_pair}"
                self.ansatz_dicts[moment][qubit_pair] = gate.name + direction[1] + f"-{qubit}"
                break

        self.convert_to_qml()
        self.draw_ansatz()