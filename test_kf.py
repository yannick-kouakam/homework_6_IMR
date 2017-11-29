from utils.utils import *
from KalmanFilter import KalmanFilter
import numpy as np

Q = np.array([[5.2,0.0,0.0],[0.0,0.983,0.0],[0.0,0.0,0.332]])
observation_covariance = np.array([[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.1]])
state_transformation = []


data = read("../hw6_data.txt")
X,X1 = process_model(data)
x = [data[0] for data in X]
y = [data[1] for data in X]

x1 = [item[0] for item in X1]
y1 = [item[1] for item in X1]
plt.plot(x,y,label='position',c='b')
plt.plot(x1,y1,label='observation',c='r')
plt.legend()
plt.show()
