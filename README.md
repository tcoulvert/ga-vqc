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

# Config (hyperparameters) for GA, see full list in example
config = {
    'backend': 'simple',
    'vqc': main,
    'vqc_config': {},
}

# Create the GA with the given hyperparameters
ga = gav.setup(config)

# Evolve the GA and search for the best ansatz
ga.evolve()
```