from abc import ABC, abstractmethod


# Currently unusable with pennylane API due to how
#  pennylane handles circuit classes (they don't).
#  Useful as a guideline for quantum circuit
#  structure and as a future-proofing in case
#  pennylane adds in classes.
class VQC(ABC):
    @abstractmethod
    def main():
        pass

    @abstractmethod
    def circuit():
        pass

    @abstractmethod
    def train():
        pass
