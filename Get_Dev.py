dev = None

def get_dev(config):
    global dev
    dev = qml.device('qulacs.simulator', wires=(2*config['n_qubits'] - config['n_latent'] + 1), shots=config['n_shots'])
    
    return dev