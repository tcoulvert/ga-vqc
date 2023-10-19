from abc import ABC, abstractmethod


class GA_Individual(ABC):
    @abstractmethod
    def generate_dicts(self):
        pass

    @abstractmethod
    def generate_qml(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def generate_vector(self):
        pass

    @abstractmethod
    def mutate(self):
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
