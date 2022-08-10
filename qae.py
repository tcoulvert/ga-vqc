import psutil
# print(f'Mem initial - {psutil.Process().memory_info().rss / (1024 * 1024)}')
import contextlib
import os
import time

import matplotlib.pyplot as plt # big mems
# from matplotlib.pyplot import figure, style, plot, ylabel, xlabel, legend, savefig
# print(f'Mem pyplot - {psutil.Process().memory_info().rss / (1024 * 1024)}')
import pennylane as qml # big mems
# print(f'Mem pennylane - {psutil.Process().memory_info().rss / (1024 * 1024)}')
from pennylane import numpy as np
from pickle import dump
from scipy import pi

from VQC_ABC import VQC

def main(ansatz, ansatz_save, params, events, train_size, n_ansatz_qubits, n_latent_qubits, rng_seed, ix, gen, start_time, n_shots):
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{ix%8}"
    # time.sleep(ix)
    # with contextlib.redirect_stdout(None):
    #     exec('import setGPU') # big mems
    # print(f'Mem setGPU - {psutil.Process().memory_info().rss / (1024 * 1024)}')
    n_trash_qubits = n_ansatz_qubits - n_latent_qubits
    n_wires = n_ansatz_qubits + n_trash_qubits + 1
    swap_pattern = compute_swap_pattern(n_ansatz_qubits, n_latent_qubits, n_trash_qubits, n_wires)
    
    dev = qml.device('qulacs.simulator', wires=n_wires, gpu=True, shots=n_shots) # big mems
    # print(f'Mem qml device - {psutil.Process().memory_info().rss / (1024 * 1024)}')
    qnode = qml.QNode(circuit, dev, diff_method='best')
    
    config = {
        'qnode': qnode,
        'ansatz': ansatz,
        'ansatz_save': ansatz_save,
        'params': params,
        'train_size': train_size,
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
    opt = qml.QNGOptimizer(1e-1, approx='block-diag')

    best_perf = [2, np.array([])]
    stop_check = [[0,0], [0,0]]
    stop_check_factor = 40
    step_size_factor = 0
    theta = pi * rng.random(size=np.shape(np.array(config['params'])), requires_grad=True)
    event_sub = rng.choice(events, config['train_size'], replace=False)
    step = 0
    while True:
        if config['train_size'] > 30:
            event_batch = rng.choice(event_sub, 30, replace=False)
        else:
            event_batch = event_sub

        grads = np.zeros((event_batch.shape[0], theta.shape[0]))
        costs = np.zeros(event_batch.shape[0])

        # iterating over all the training data
        for i in range(event_batch.shape[0]):
            # fub_stud = qml.metric_tensor(circuit, approx="block-diag")(theta, event=event_batch[i], config=config)
            # grads[i] = np.matmul(fub_stud, opt.compute_grad(circuit, (theta, event_batch[i], config), {})[0][0])
            fub_stud = qml.metric_tensor(config['qnode'], approx="block-diag")(theta, event=event_batch[i], config=config)
            grads[i] = np.matmul(fub_stud, opt.compute_grad(config['qnode'], (theta, event_batch[i], config), {})[0][0])
            costs[i] = circuit(theta, event=event_batch[i], config=config)
        if best_perf[0] > costs.mean(axis=0):
            best_perf[0] = costs.mean(axis=0)
            best_perf[1] = theta
        theta = theta - (10**step_size_factor * np.sum(grads, axis=0))
        qng_cost.append(costs.mean(axis=0))

        # checking the stopping condition
        if step > stop_check_factor:
            stop_check[0][0] = np.mean(qng_cost[-40:-20], axis=0)
            stop_check[0][1] = np.std(qng_cost[-40:-20], axis=0)
            stop_check[1][0] = np.mean(qng_cost[-20:], axis=0)
            stop_check[1][1] = np.std(qng_cost[-20:], axis=0)
            if np.isclose(stop_check[0][0], stop_check[1][0], atol=np.amin([stop_check[0][1], stop_check[1][1]])):
                step_size_factor -= 1
                stop_check_factor = step + 20
                if step_size_factor < -5:
                    break
        step += 1
    
    # big mems
    # print(f'Mem done training - {psutil.Process().memory_info().rss / (1024 * 1024)}')
    script_path = os.path.dirname(os.path.realpath(__file__))
    destdir = os.path.join(script_path, 'qae_runs_%s' % config['start_time'])
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    destdir_thetas = os.path.join(destdir, 'opt_thetas')
    if not os.path.exists(destdir_thetas):
        os.makedirs(destdir_thetas)
    filepath_thetas = os.path.join(destdir_thetas, '%02d_%03dga_best%.e_data_theta' % (config['ix'], config['gen'], config['train_size']))
    np.save(filepath_thetas, best_perf[1])
    
    destdir_ansatz = os.path.join(destdir, 'opt_ansatz')
    if not os.path.exists(destdir_ansatz):
        os.makedirs(destdir_ansatz)
    filepath_ansatz = os.path.join(destdir_ansatz, '%02d_%03dga_best%.e_data_ansatz' % (config['ix'], config['gen'], config['train_size']))
    with open(filepath_ansatz, "wb") as f:
        dump(config['ansatz_save'], f)
    filepath_draw = os.path.join(destdir_ansatz, '%02d_%03dga_best%.e_draw_ansatz' % (config['ix'], config['gen'], config['train_size']))
    ansatz_draw = qml.draw(config['qnode'], decimals=None, expansion_strategy='device')(theta, event=event_batch[0], config=config)
    with open(filepath_draw, "w") as f:
        f.write(ansatz_draw)
        
    destdir_curves = os.path.join(destdir, 'qml_curves')
    if not os.path.exists(destdir_curves):
        os.makedirs(destdir_curves)
    filepath_curves = os.path.join(destdir_curves, "%02d_%03dga_QNG_Descent-%d_data.pdf" % (config['ix'], config['gen'], config['train_size']))
    plt.figure(config['train_size'])
    plt.style.use("seaborn")
    plt.plot(qng_cost, "g", label="QNG Descent - %d data" % config['train_size'])
    plt.ylabel("1 - Fid.")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig(filepath_curves)

    # big mems
    # print(f'Mem saved files - {psutil.Process().memory_info().rss / (1024 * 1024)}')
    return 1-(best_perf[0]+np.mean(qng_cost, axis=0))