# GA for VQC Ansatz Search
This is a module to support Variational Quantum Circuits by optimizing the ansatz. The ansatz optimization is performed using a Genetic Algorithm, which can be sped up with GPUs.


## Installation
Run the following to install:
```bash
$ pip install ga-vqc
```

## Contributors
This module was developed through the Caltech SURF program. Special thanks
to my mentor at Caltech.
- Jean-Roch (California Institute of Technology, Pasadena, CA 91125, USA)

## Usage
```python
import ga_vqc as gav

## Config (hyperparameters) for GA, see full list in example ##

vqc_main = <'Function that handles running your VQC optimization'>

# Example of allowed optimization gates, see Genepool.py for documentation
gates_dict = {"I": (1, 0), "RX": (1, 1), "CNOT": (2, 0)}
gates_probs = [0.35, 0.35, 0.3]
genepool = gav.Genepool(gates_dict, gates_probs)

vqc_config = {
    'num_qubits': 3,
    'etc': <'whatever config params your VQC model requires'>
}

ga_output_path = FILEPATH_FOR_GA_OUTPUT

config = gav.Config(vqc_main, vqc_config, genepool, ga_output_path)

# Create the GA with the given hyperparameters
ga = gav.setup(config)

# Evolve the GA and search for the best ansatz
ga.evolve()
```