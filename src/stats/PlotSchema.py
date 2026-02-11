from abc import ABC, abstractmethod
import matplotlib as plt


class PlotSchema:
    def __init__():
        pass

    def plot():
        pass


class PlotAction(ABC):
    @abstractmethod
    def plot(data, fig, axes):
        pass
