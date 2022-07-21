import contextlib
import time

import matplotlib.pyplot as plt
import pennylane as qml
import pickle
import scipy as sp
from math import isclose
from pennylane import numpy as np

from VQC_ABC import VQC

def main(ansatz, ansatz_save, params, events, n_ansatz_qubits, n_latent_qubits, rng_seed, ix, n_shots=5000):
    rng = np.random.default_rng(seed=rng_seed)
    time.sleep(5*rng.random())
    with contextlib.redirect_stdout(None):
        exec('import setGPU')
    n_trash_qubits = n_ansatz_qubits - n_latent_qubits
    n_wires = n_ansatz_qubits + n_trash_qubits + 1
    swap_pattern = compute_swap_pattern(n_ansatz_qubits, n_latent_qubits, n_trash_qubits, n_wires)
    
    dev = qml.device('qulacs.simulator', wires=n_wires, gpu=True, shots=n_shots)
    qnode = qml.QNode(circuit, dev, diff_method='best')
    
    config = {
        'qnode': qnode,
        'ansatz': ansatz,
        'ansatz_save': ansatz_save,
        'params': params,
        'n_latent_qubits': n_latent_qubits,
        'n_trash_qubits': n_trash_qubits,
        'n_wires': n_wires,
        'swap_pattern': swap_pattern,
        'rng_seed': rng_seed,
        'ix': ix
    }
    
    best_fid = train(events, config)
    return best_fid

def compute_swap_pattern(n_ansatz_qubits, n_latent_qubits, n_trash_qubits, n_wires):
    swap_pattern = []
    for i in range(n_trash_qubits):
        single_swap = [n_wires-1, 0, 0]
        single_swap[1] = n_latent_qubits + i
        single_swap[2] = n_ansatz_qubits + i

        swap_pattern.append(single_swap)
        
    return swap_pattern

# @qml.qnode(dev, diff_method='best')
def circuit(params, event=None, config=None):
    # Embed the data into the circuit
    qml.broadcast(qml.RX, wires=range(config['n_latent_qubits']+config['n_trash_qubits']), pattern='single', parameters=event)
    qml.Hadamard(wires=config['n_wires']-1)

    # Run the actual circuit ansatz
    for m in config['ansatz']:
        exec(m)

    # Perform the SWAP-Test for a qubit fidelity measurement
    qml.broadcast(qml.CSWAP, wires=range(config['n_latent_qubits'], config['n_wires']), pattern=config['swap_pattern'])
    qml.Hadamard(wires=config['n_wires']-1)

    return qml.expval(qml.PauliZ(wires=config['n_wires']-1))

def train(events, config):
    circuit = config['qnode']
    rng = np.random.default_rng(seed=config['rng_seed'])
    qng_cost = [1]
    opt = qml.QNGOptimizer(1e-6, approx='block-diag')

#         for train_size in [1, 10, 100, 1000, 10000]:
#             step = 0
#             best_perf = [2, np.zeros(3)]
#             mean_check = [0]
#             theta = sp.pi * self.rng.random(size=3, requires_grad=True)
#             event_sub = self.rng.choice(events, train_size, replace=False)
#             while True:
#                 if train_size > 30:
#                     event_batch = self.rng.choice(event_sub, 30, replace=False)
#                 else:
#                     event_batch = event_sub

#                 grads = np.zeros((event_batch.shape[0], theta.shape[0]))
#                 costs = np.zeros(event_batch.shape[0])

#                 # if step%25 == 1:
#                 #     print('Step %d with cost %.4f' % (step, qng_cost[ix][-1]))

#                 for i in range(event_batch.shape[0]):
#                     fub_stud = qml.metric_tensor(circuit, approx='block-diag')(theta, event=event_batch[i]) # doesnt compute fubiny-study automatically??
#                     grads[i] = np.matmul(fub_stud, opt.compute_grad(circuit, (theta, event_batch[i]), {})[0][0]) # isn't already anti-gradient??
#                     costs[i] = circuit(theta, event=event_batch[i])
#                 if best_perf[0] > costs.mean(axis=0):
#                     best_perf[0] = costs.mean(axis=0)
#                     best_perf[1] = theta
#                     print(best_perf)
#                 theta = theta - np.sum(grads, axis=0)
#                 qng_cost[ix].append(costs.mean(axis=0))
#                 mean_check[ix] += costs.mean(axis=0)

#                 if np.nonzero(np.isclose(np.sum(grads, axis=0), np.zeros(theta.shape[0]), rtol=1e-05, atol=1e-08, equal_nan=False))[0].shape == 0:
#                     break
#                 elif step%1000 == 0:
#                     mean_check[ix] = mean_check[ix] / 1000
#                     if step%2000 == 0 and isclose(mean_check[-2], mean_check[-1], rtol=1e-03, atol=1e-05):
#                         break
#                     mean_check.append(0)
#                 step += 1

    train_size, steps = 1, 100
    best_perf = [2, np.array([])]
    theta = sp.pi * rng.random(size=np.shape(np.array(config['params'])), requires_grad=True)
    event_sub = rng.choice(events, train_size, replace=False)
    for _ in range(steps):

        if train_size > 30:
            event_batch = rng.choice(event_sub, 30, replace=False)
        else:
            event_batch = event_sub

        grads = np.zeros((event_batch.shape[0], theta.shape[0]))
        costs = np.zeros(event_batch.shape[0])

        if _ == (steps-1):
            print('Step %d with cost %.4f' % (_, qng_cost[-1]))
        # print('Step %d with cost %.4f' % (_, qng_cost[-1]))

        for i in range(event_batch.shape[0]):
            # print(qml.draw(circuit, expansion_strategy='device', show_all_wires=True)(theta, event=event_batch[i], config=config))
            fub_stud = qml.metric_tensor(circuit, approx="block-diag")(theta, event=event_batch[i], config=config)
            grads[i] = np.matmul(fub_stud, opt.compute_grad(circuit, (theta, event_batch[i], config), {})[0][0])
            costs[i] = circuit(theta, event=event_batch[i], config=config)
        if best_perf[0] > costs.mean(axis=0):
            best_perf[0] = costs.mean(axis=0)
            best_perf[1] = theta
        theta = theta - np.sum(grads, axis=0)
        qng_cost.append(costs.mean(axis=0))

    np.save('%02dga_best%.e_data_theta' % (config['ix'], train_size), best_perf[1])
    with open('%02dga_best%.e_data_ansatz' % (config['ix'], train_size), "wb") as f:
        pickle.dump(config['ansatz_save'], f)
    plt.figure(train_size)
    plt.style.use("seaborn")
    plt.plot(qng_cost, "g", label="QNG Descent - %d data" % train_size)
    plt.ylabel("1 - Fid.")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig("%02dga_QNG_Descent-%d_data.pdf" % (config['ix'], train_size))
    # plt.show()

    return 1-(best_perf[0]+np.mean(qng_cost, axis=0))