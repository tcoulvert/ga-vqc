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

    def __init__(self, n_qubits, n_moments, n_vector_moments, genepool, rng_seed, dicts=None, diagram=None):
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
        self.n_vector_moments = n_vector_moments
        self.genepool = genepool
        self.rng = np.random.default_rng(seed=rng_seed)
        self.pennylane_rng = pnp.random.default_rng(seed=rng_seed)

        self.dicts = []          # Used to perform GA optimization on
        self.qml = []            # Used to run the actual VQC (list of strings of python code to run VQC on pennylane)
        self.diagram = str()     # Used to visualize the VQC 
                                    # and compile the circuit through the draw function
                                    # and (can be used to) track diversity through string distance
        self.vector = []         # (Can be) used to keep track of which circuits have been run and distance between circuits
        self.params = []

        if dicts is None and diagram is None:
            self.generate_dicts()
        elif dicts is not None:
            self.generate_from_dicts(dicts)
        elif diagram is not None:
            self.generate_from_diagram(diagram)

    def __len__(self):
        return len(self.dicts[0])

    def __getitem__(self, key):
        if type(key) is tuple:
            return self.dicts[int(key[0])][int(key[1])]
        else:
            return self.dicts[int(key)]

    def __setitem__(self, key, value):
        """
        Allows editing of the ansatz via the [] operators.
        
        TODO: figure out how to allow automatic updates
        """
        if type(key) is tuple:
            self.dicts[int(key[0])][int(key[1])] = value
        else:
            self.dicts[int(key)] = value
        self.update()

    def __str__(self):
        return self.diagram

    def __repr__(self):
        return self.__str__()
    
    def update(self, FULL=True):
        if FULL:
            self.generate_qml()
            self.draw()
        self.compile()
        self.generate_qml()
        self.generate_vector(self.n_vector_moments)

    def overwrite(self, moment, swapset, new_moment_dict):
        for qubit in swapset:
            self.dicts[moment][qubit] = new_moment_dict[qubit]
        self.update()

    def generate_dicts(self):
        """
        Generates the ansatz stochastically based on the other member variables.
        """
        for moment in range(self.n_moments):
            self.dicts.append(dict.fromkeys(range(self.n_qubits), 0))
            qubits = self.rng.permutation(self.n_qubits)  # which qubit to pick first
            for qubit in qubits:
                if self.dicts[moment][qubit] != 0:
                    continue
                gate = self.genepool.choice()

                if gate.n_qubits == 1:
                    self.dicts[moment][qubit] = gate.name
                    continue
                elif gate.n_qubits == 2:
                    qubit_pairs = self.rng.permutation(
                        self.n_qubits
                    )
                    for qubit_pair in qubit_pairs:
                        if self.dicts[moment][qubit_pair] != 0 or qubit_pair == qubit:
                            continue

                        direction = self.rng.permutation(["_C", "_T"])
                        self.dicts[moment][qubit] = gate.name + direction[0] + f"-{qubit_pair}"
                        self.dicts[moment][qubit_pair] = gate.name + direction[1] + f"-{qubit}"
                        break

                    if self.dicts[moment][qubit] == 0:
                        gate = self.genepool.choice(n_qubits=1)
                        self.dicts[moment][qubit] = gate.name
                else:
                    raise Exception("Gates with more than 2 qubits haven't been implemented yet.")
                
        self.update()

    def generate_from_dicts(self, dicts):
        self.dicts = copy.deepcopy(dicts)
        self.update()

    def generate_from_diagram(self, diagram):
        self.diagram = copy.deepcopy(diagram)
        self.update(FULL=False)

    def generate_qml(self):
        """
        Converts the ansatz into a general format for the QML.

            moment_dict example: _C (_T) means current qubit is control (target) qubit of 2-qubit gate
                {'I': [],
                 'RX': [],
                 'RY': [],
                 'RZ': [],
                 'CNOT': []}
        """
        self.qml = []
        self.params = []
        for moment in range(len(self.dicts)):
            moment_dict = {gate.name: list() for gate in self.genepool.gates}
            stored_i = []
            for qubit in range(self.n_qubits):
                _ix = self.dicts[moment][qubit].find("_")
                if _ix < 0:
                    moment_dict[self.dicts[moment][qubit]].append(qubit)
                else:
                    if qubit in stored_i:
                        continue
                    _1ix = self.dicts[moment][qubit][_ix + 1]
                    q_p = int(self.dicts[moment][qubit][-1])
                    stored_i.append(q_p)
                    if _1ix == "C":
                        moment_dict[self.dicts[moment][qubit][:_ix]].append([qubit, int(q_p)])
                    else:
                        moment_dict[self.dicts[moment][qubit][:_ix]].append([int(q_p), qubit])
            
            for gate_name in moment_dict.keys():
                if len(moment_dict[gate_name]) == 0 or gate_name == "I":
                    continue
                if self.genepool.n_qubits(gate_name) == 1: # Change to check n_params of gate
                    if self.genepool.n_params(gate_name) > 0:
                        self.qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={moment_dict[gate_name]}, pattern='single', " +
                            f"parameters=params[{len(self.params)}:{len(self.params) + len(moment_dict[gate_name])}])"
                        )
                        for _ in range(len(moment_dict[gate_name])):
                            # self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name), requires_grad=True))
                            self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name)))
                    else:
                        self.qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={moment_dict[gate_name]}, pattern='single')"
                        )
                elif self.genepool.n_qubits(gate_name) == 2:
                    if self.genepool.n_params(gate_name) > 0:
                        self.qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={np.array(moment_dict[gate_name]).flatten(order='C').tolist()}, " +
                            f"parameters=params[{len(self.params)}:{len(self.params) + len(moment_dict[gate_name])}])" +
                            f"pattern={moment_dict[gate_name]})"
                        )
                        for _ in range(len(moment_dict[gate_name])):
                            # self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name), requires_grad=True))
                            self.params.append(np.pi * self.pennylane_rng.random(size=self.genepool.n_params(gate_name)))
                    else:
                        self.qml.append(
                            f"qml.broadcast(qml.{gate_name}, wires={np.array(moment_dict[gate_name]).flatten(order='C').tolist()}, " +
                            f"pattern={moment_dict[gate_name]})"
                        )
        self.params = pnp.array(self.params, dtype=object, requires_grad=True)

    def circuit(self, params, event=None):
        for m in self.qml:
            exec(m)
        return qml.expval(qml.PauliZ(wires=[self.n_qubits - 1]))

    def draw(self):
        qnode = qml.QNode(
                qml.compile(basis_set=self.genepool.gate_list(), num_passes=3)(self.circuit),
                qml.device("default.qubit", wires=self.n_qubits, shots=1),
            )
        self.diagram = qml.draw(
            qnode,
            decimals=None,
            max_length=1e6,
            expansion_strategy="device",
            show_all_wires=True,
        )(self.params, event=[i for i in range(1, self.n_qubits+1)])[:-3]

    def compile(self):
        """
        Update the ansatz from the drawn ansatz after compiling the circuit. 

        TODO: Figure out how to make this general for any gate (or at 
                minimum any gate the compiler spits out) -> edit specific logic to use genepool
        """
        self.dicts = []

        ix_arr = [(None, 0)] + [(bar.start()+1, n.end()) for bar, n in zip(re.finditer('┤', self.diagram), re.finditer('\n', self.diagram))] + [(-2, None)]
        qubit_array = [self.diagram[ix_arr[i][1] : ix_arr[i+1][0]] for i in range(len(ix_arr) - 1)]

        def find_other_qubit(qubit_array, qubit_ix, gate_ix):
            other_qubit = -1
            for q_ix in range(qubit_ix+1, len(qubit_array)):
                if qubit_array[q_ix][gate_ix] == '╰':
                    other_qubit = q_ix
                    break

            return other_qubit
        
        def moment_array(qubit_array):
            qubit_moments = dict.fromkeys(range(self.n_qubits), -1)
            moment = 0
            gate_flag = False
            for string_ix in range(3, len(qubit_array[0])):     # 3 b/c indeces 0-2 are where pennylane print qubit number like '0: ' and then at index 3 the circuit drawing begins
                dash_count = 0
                for q_ix in range(len(qubit_array)):
                    if qubit_array[q_ix][string_ix] == '─' or qubit_array[q_ix][string_ix] == '┤':
                        dash_count += 1
                    else:
                        if isinstance(qubit_moments[q_ix], int):
                            qubit_moments[q_ix] = []
                        elif moment in qubit_moments[q_ix]:
                            continue
                        qubit_moments[q_ix].append(moment)
                if dash_count == len(qubit_array):
                    if gate_flag:
                        moment += 1
                    gate_flag = False
                else:
                    gate_flag = True

            return qubit_moments, moment

        qubit_moments, n_moments = moment_array(qubit_array)
        for _ in range(n_moments):
            self.dicts.append(dict.fromkeys(range(self.n_qubits), 0))
        qubit_ix = 0
        for qubit in qubit_array:
            gate_ix = 0
            dash_flag = False
            for str_ix in range(3, len(qubit)):
                # THIS LOGIC SPECIFIC TO ONE GENEPOOL (RX, RY, RZ, Rϕ, CNOT)
                #  -> change to just search over self.genepool.gates array for matching sequence
                if qubit[str_ix] == '─':
                    dash_flag = True
                    continue

                if qubit[str_ix] == 'R':
                    moment = qubit_moments[qubit_ix][gate_ix]

                    if qubit[str_ix+1] == 'ϕ':
                        self.dicts[moment][qubit_ix] = 'PhaseShift' # Is there a way to get qml.draw() to use the actual gate names?
                    else:
                        self.dicts[moment][qubit_ix] = 'R' + qubit[str_ix+1]
                elif qubit[str_ix] == '╭':
                    moment = qubit_moments[qubit_ix][gate_ix]

                    other_qubit_ix = find_other_qubit(qubit_array, qubit_ix, str_ix)
                    if qubit[str_ix+1] == 'X':
                        self.dicts[moment][qubit_ix] = 'CNOT_T-' + str(other_qubit_ix)
                        self.dicts[moment][other_qubit_ix] = 'CNOT_C-' + str(qubit_ix)
                    else:       # qubit[j+1] == '●'
                        self.dicts[moment][qubit_ix] = 'CNOT_C-' + str(other_qubit_ix)
                        self.dicts[moment][other_qubit_ix] = 'CNOT_T-' + str(qubit_ix)
                elif not dash_flag:
                    continue

                gate_ix += 1
                dash_flag = False
            
            qubit_ix += 1
            
        for moment_dict in self.dicts:
            for k, v in moment_dict.items():
                if v == 0:
                    moment_dict[k] = 'I'

        self.n_moments = len(self.dicts)
        if self.n_moments > self.n_vector_moments:
            self.n_vector_moments = 2 * self.n_moments

    def add_moment(self, method='random', **kwargs):
        """
        TODO: change to randomly generate new moment?
        """
        moment = self.rng.integers(self.n_moments)

        if method == 'random':
            raise Exception("Method not yet supported.")
        elif method == 'duplicate':
            self.dicts.append(copy.deepcopy(self.dicts[moment]))
            self.n_moments += 1
        elif method == 'pad':
            for _ in range(kwargs['num_pad']):
                self.dicts.append(dict.fromkeys(range(self.n_qubits), 'I'))
                self.n_moments += 1
        else:
            raise Exception("Method not supported.")
        
        self.update()

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

        if self.dicts[moment][qubit].find("_") > 0:
            double_swap_flag += 1
            gate_pair = self.genepool.choice(n_qubits=1)
            self.dicts[moment][int(self.dicts[moment][qubit][-1])] = gate_pair.name

        if gate.n_qubits == 1:
            self.dicts[moment][qubit] = gate.name
        else:
            qubit_pairs = self.rng.permutation(self.n_qubits)
            for qubit_pair in qubit_pairs:
                if qubit_pair == qubit:
                    continue
                if self.dicts[moment][qubit_pair].find("_") > 0:
                    double_swap_flag += 1
                    gate_pair = self.genepool.choice()
                    if double_swap_flag == 2 and gate_pair.n_qubits == 2:
                        direction = self.rng.permutation(["_C", "_T"])
                        self.dicts[moment][int(self.dicts[moment][qubit][-1])] = (
                            gate_pair.name + direction[0] + f"-{int(self.dicts[moment][qubit_pair][-1])}"
                        )
                        self.dicts[moment][int(self.dicts[moment][qubit_pair][-1])] = (
                            gate_pair.name + direction[1] + f"-{int(self.dicts[moment][qubit][-1])}"
                        )
                    else:
                        gate_pair = self.genepool.choice(n_qubits=1)
                        self.dicts[moment][int(self.dicts[moment][qubit_pair][-1])] = gate_pair.name

                direction = self.rng.permutation(["_C", "_T"])
                self.dicts[moment][qubit] = gate.name + direction[0] + f"-{qubit_pair}"
                self.dicts[moment][qubit_pair] = gate.name + direction[1] + f"-{qubit}"
                break

        self.update()

    def generate_vector(self, max_moments):
        """
        TODO: Change pad from affecting circuit to happeneing automatically here
        """
        self.n_vector_moments = max_moments
        vector = []

        ### single-qubit gates ###
        for moment in range(max_moments):
            for qubit in range(self.n_qubits):
                one_qubit_states = []
                for _ in range(
                    self.genepool.n_gates(
                        search_param={'n_qubits': 1}
                    )
                ):
                    one_qubit_states.extend([0])

                if moment >= self.n_moments or self.genepool.n_qubits(self.dicts[moment][qubit]) != 1:
                    vector.extend([i for i in one_qubit_states])
                    continue
                one_qubit_states[self.genepool.index_of(self.dicts[moment][qubit])] = 1 # Assumes 'I' always in index 0, and cannot NOT include 'I'
                vector.extend([i for i in one_qubit_states])

        vector.append(-1)

        ### 2-qubit gates ###
        for moment in range(max_moments):
            # [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1), ('I', 'I')]
            two_qubit_states = []
            for _ in range(
                self.genepool.n_gates(
                    search_param={'n_qubits': 2}
                )
            ):
                two_qubit_states.extend([0 for __ in range(np.math.factorial(self.n_qubits) + 1)])
                two_qubit_states[-1] = 1

            for qubit in range(self.n_qubits):
                if moment >= self.n_moments:
                    break
                if self.dicts[moment][qubit].find('_') > 0: # Doesn't work for passing more than 1 2-qubit gate, and only works for 'control'/'target' gates
                    two_qubit_states[-1] = 0
                    if self.dicts[moment][qubit][-3] == 'C':
                        two_qubit_states[2*qubit] = 1
                    elif self.dicts[moment][qubit][-3] == 'T':
                        two_qubit_states[2*qubit + 1] = 1
                    break
            vector.extend(two_qubit_states)

        ### 3+ qubit gates ###
        # TO DO

        self.vector = vector