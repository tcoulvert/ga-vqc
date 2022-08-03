import contextlib
import os
import time

import matplotlib.pyplot as plt
import pennylane as qml
import pickle
import scipy as sp
from math import isclose
from numba import cuda
from pennylane import numpy as np

from VQC_ABC import VQC

def main(ansatz, ansatz_save, params, events, n_ansatz_qubits, n_latent_qubits, rng_seed, ix, gen, start_time, n_shots=5000):
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    time.sleep(ix)
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
        'ix': ix,
        'gen': gen,
        'start_time': start_time,
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
    opt = qml.QNGOptimizer(1e-5, approx='block-diag')

    train_size, steps = 10, 200
    best_perf = [2, np.array([])]
    stop_check = [[0,0], [0,0]]
    theta = sp.pi * rng.random(size=np.shape(np.array(config['params'])), requires_grad=True)
    event_sub = rng.choice(events, train_size, replace=False)
    step = 0
    while True:
        if train_size > 30:
            event_batch = rng.choice(event_sub, 30, replace=False)
        else:
            event_batch = event_sub

        grads = np.zeros((event_batch.shape[0], theta.shape[0]))
        costs = np.zeros(event_batch.shape[0])

        # iterating over all the training data
        for i in range(event_batch.shape[0]):
            fub_stud = qml.metric_tensor(circuit, approx="block-diag")(theta, event=event_batch[i], config=config)
            grads[i] = np.matmul(fub_stud, opt.compute_grad(circuit, (theta, event_batch[i], config), {})[0][0])
            costs[i] = circuit(theta, event=event_batch[i], config=config)
        if best_perf[0] > costs.mean(axis=0):
            best_perf[0] = costs.mean(axis=0)
            best_perf[1] = theta
        theta = theta - np.sum(grads, axis=0)
        qng_cost.append(costs.mean(axis=0))

        # checking the stopping condition
        if step%30 == 0:
            if step%60 == 0:
                stop_check[1][0] = np.mean(qng_cost[step-30:], axis=0)
                stop_check[1][1] = np.std(qng_cost[step-30:], axis=0)
                if np.isclose(stop_check[0][0], stop_check[1][0], atol=np.amin([stop_check[0][1], stop_check[1][1]])):
                    break
            stop_check[0][0] = np.mean(qng_cost[step-30:], axis=0)
            stop_check[0][1] = np.std(qng_cost[step-30:], axis=0)
        step += 1

    script_path = os.path.dirname(os.path.realpath(__file__))
    destdir = os.path.join(script_path, 'qae_runs_%s' % config['start_time'])
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    destdir_thetas = os.path.join(destdir, 'opt_thetas')
    if not os.path.exists(destdir_thetas):
        os.makedirs(destdir_thetas)
    filepath_thetas = os.path.join(destdir_thetas, '%02d_%03dga_best%.e_data_theta' % (config['ix'], config['gen'], train_size))
    np.save(filepath_thetas, best_perf[1])
    
    destdir_ansatz = os.path.join(destdir, 'opt_ansatz')
    if not os.path.exists(destdir_ansatz):
        os.makedirs(destdir_ansatz)
    filepath_ansatz = os.path.join(destdir_ansatz, '%02d_%03dga_best%.e_data_ansatz' % (config['ix'], config['gen'], train_size))
    with open(filepath_ansatz, "wb") as f:
        pickle.dump(config['ansatz_save'], f)
        
    destdir_curves = os.path.join(destdir, 'qml_curves')
    if not os.path.exists(destdir_curves):
        os.makedirs(destdir_curves)
    filepath_curves = os.path.join(destdir_curves, "%02d_%03dga_QNG_Descent-%d_data.pdf" % (config['ix'], config['gen'], train_size))
    plt.figure(train_size)
    plt.style.use("seaborn")
    plt.plot(qng_cost, "g", label="QNG Descent - %d data" % train_size)
    plt.ylabel("1 - Fid.")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig(filepath_curves)
    # plt.show()

    return 1-(best_perf[0]+np.mean(qng_cost, axis=0))