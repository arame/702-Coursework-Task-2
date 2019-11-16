import numpy as np

class Derivative:
    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)
    
    @staticmethod
    def reLU(arr):
        arr1 = np.where(arr <= 0, 0, 1)
        return arr1
        #for i in range(len(arr)):
        #    item = np.float64(arr[i])
        #    if not isinstance(item, float):
        #        raise TypeError("Only floats are allowed")
            #number = Derivative.check_number(arr[i])
        #    if item <= 0:
        #        arr[i] = 0
        #    else:
        #        arr[i] = 1
        #return arr

    @staticmethod
    def check_number(item):
        try:
            return int(item)
        except ValueError:
            print("{0} is not an number!".format(item))
            return 0
        except OverflowError:
            print("{0} cannot be infinity!".format(item))
            return 0
