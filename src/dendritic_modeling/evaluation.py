from dendritic_modeling import logger



class Evaluation:
    """
    Evaluation class for evaluating the model
    """
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def evaluate(self):
        pass


class SimpleSum:
    """
    ComplexSum class for summing numbers
    """

    def __init__(self, numbers):
        self.numbers = numbers

    def sum(self):
        logger.info('Summing numbers')
        logger.debug(f'Numbers: {self.numbers}')
        return sum(self.numbers)