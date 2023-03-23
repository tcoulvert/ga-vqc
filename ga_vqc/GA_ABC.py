from abc import ABC, abstractmethod


class GA_Individual(ABC):
    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def convert_to_qml(self):
        pass


class GA_Model(ABC):
    @abstractmethod
    def evolve(self):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def select(self):
        pass

    @abstractmethod
    def mate(self):
        pass

    @abstractmethod
    def evaluate_fitness(self):
        pass
