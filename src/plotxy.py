import matplotlib.pyplot as plt

class plotxy:

    @staticmethod
    def standard( x, y, x_axis, y_axis):
        plt.plot(x, y)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()