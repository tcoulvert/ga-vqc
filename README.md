# GA for VQC Ansatz Search
Quantum Adiabatic Machine Learning with Zooming IMproved. This is a supervised ML algorithm used to train a Binary Classifier on D-Wave's Quantum Annealers. The library has been set up to be compatible with Scikit-Learn's data representation. The algortihm is intended to be generalizable to any Binary ML problem.

In order to run the program you'll need D-Wave credentials, these can be obtained at https://cloud.dwavesys.com/leap/signup/. You'll need a github account in order to sign up. This account will give you the "endpoint_url" and "account_token" referenced below.

## Installation
Run the following to install:
```bash
$ pip install qamlzim
```

## Contributors
Special thanks to everyone who helped me develop this module
- My Mentor:
    - Jean-Roch (California Institute of Technology, Pasadena, CA 91125, USA)
- All of QMLQCF, with special mentions of:
    - Jean-Roch (California Institute of Technology, Pasadena, CA 91125, USA)
    - Daniel Lidar (University of Southern California, Los Angeles, CA 90007, USA)
    - Gabriel Perdue (Fermi National Accelerator Laboratory, Batavia, IL 60510, USA)
- Author of the QAE code:
    - Alexander Zlokapa (Massachusetts Institute of Technology, Cambridge, MA 02139, USA)
- Mentoring for code practices:
    - Otto Sievert (GoPro, Inc.)

## Usage
```python
import qamlzim

# Generate the Environment (Data) for the Model
env = qamlzim.TrainEnv(X_train, y_train, endpoint_url, account_token)

# Generate the Config (Hyperparameters) for the Model
config = qamlzim.ModelConfig()

# Create the Model and begin training
model = qamlzim.Model(config, env)
model.train()
```

## Developing QAML-ZIM
To install qamlzim, along with the tools you need to develop and run tests, run the following in your virtualenv:
```bash
$ pip install -e .[dev]
```