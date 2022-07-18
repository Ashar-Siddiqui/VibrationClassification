import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
from os.path import dirname, join

def display_col():
    data = np.genfromtxt(join(dirname(__file__), 'test.csv'), delimiter=',', skip_header=1, dtype=np.float32)

    input_voltage = data[:, 0]
    vibration_1 = data[:, 2]
    vibration_2 = data[:, 3]
    vibration_3 = data[:, 4]

    #accessing columns
    #print(data[:, 2])
    #accessing rows
    #print(data[3:])
    #print(data)

if __name__ == '__main__':
    display_col()
