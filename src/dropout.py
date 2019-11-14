import numpy as np

class Dropout:
    filter_percentage = 0.25     # ie 50%  dropout

    @staticmethod
    def get_mask(x):

        dropout_mask = (np.random.rand(*x.shape) < Dropout.filter_percentage) / Dropout.filter_percentage
        return dropout_mask