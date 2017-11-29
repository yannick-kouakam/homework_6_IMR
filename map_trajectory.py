#!/usr/bin/python
from math import cos, sin
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.stats
from scipy.stats import norm, multivariate_normal
import math
from random import random

def diff(data,optional=None):
    if optional is not None:
        for i in range(0,len(data)):
            data[i] -= optional[i]
    else:
        for i in range(1,len(data)):
            data[i-1] = data[i]-data[i-1]

    return data

def summ(data1,data2):
    for i in range(0,len(data1)):
        data1[i]+=data2[i]
    return data1

def calculate_position_fromsensordata(data):
    pass


def read(path):

    file = open(path,'r')
    lines = file.readlines()
    data = []

    for line in lines:
         line = line[0:len(line)-2]
         line = line.split(",")
         data.append([float(x) for x in line[0:len(line)]])

    return data



def process_model(sensordata):
    current_pos = [0,0,0]
    previous_data = [0,0,0,0,0,0]
    trajectory =[]
    leftwall= []
    rightwall=[]
    for data in sensordata:
        dsl = data[1] - previous_data[1]
        dsr = data[2] - previous_data[2]
        teta=math.radians(current_pos[2]+(dsr-dsl)/(2*0.12))

        u = [((dsl+dsr)/2)*cos(teta), ((dsl+dsr)/2)*sin(teta),(dsr-dsl)/(0.12)]

        upd_pos = np.sum([current_pos,u],axis=0)
        current_pos=upd_pos
        trajectory.append(upd_pos)
        # left wall coordinate
        theta = math.radians(data[5])
        xlw =current_pos[0] + data[3]*cos(theta - math.pi/2)
        ylw =current_pos[1] + data[3]*cos(theta - math.pi/2)
        leftwall.append([xlw,ylw])
        # right wall coordinate
        xlw =current_pos[0]- data[4]*cos(theta + math.pi/2)
        ylw =current_pos[1] -data[4]*cos(theta + math.pi/2)
        rightwall.append([xlw,ylw])
        previous_data = data

    return( np.array(trajectory), np.array(leftwall), np.array(rightwall))


def get_position(data):
    r = 0.028
    d = 0.149
    x_cord=[0]
    y_cord=[0]
    theta =[0]

    leftWheel = [x[1] for x in data]
    rigthtWheel = [x[2] for x in data]

    for i in range(1,len(data)):
        dl = (leftWheel[i] -leftWheel[i-1])
        dr = (rigthtWheel[i] -rigthtWheel[i-1])
        temp = theta[i-1]+(dr-dl)*r/d

        theta.append(temp)
        x_cord.append(x_cord[i-1] +r/2*cos(math.radians(theta[i-1])) )
        y_cord.append(y_cord[i-1] +r/2*sin(math.radians(theta[i-1])) )

    return (x_cord,y_cord,theta)



def buildmap(sensordata,x,y,theta):
    distL=[x[3] for x in sensordata]
    distR=[x[4] for x in sensordata]
    left_features=[]
    right_features=[]
    x_coord=0
    y_coord=0

    for  i in range(0,len(x)):
        #find coordinates of the features on the left
        x_coord = x[i] + distL[i]*cos(math.radians(theta[i]-90))
        y_coord = y[i] + distL[i]*cos(math.radians(theta[i]-90))
        left_features.append([x_coord,y_coord])

        # find the coordinates of the feature on the right
        x_coord = x[i] - distR[i]*cos(math.radians(theta[i]+90))
        y_coord = y[i] - distR[i]*cos(math.radians(theta[i]+90))
        right_features.append([x_coord,y_coord])

    return (left_features,right_features)



class GaussianDistribution(object):
    """docstring for gaussianDistribution."""

    def __init__(self, mean:np.array,covariance:np.array):

        self._mean = mean
        self._covariance = covariance

    @property
    def covariance(self):
        return self._covariance

    @property
    def mean(self):
        return self._mean



    def to_cov(X,n):
        """
        checks if X is a scalar in what case it returns a covariance
        matrix generated from it as the identity matrix mutiplied
        by X. the dimension will be n*n.
        If X is already a numpy array then it is returned unchanged.
        """

        try:
            X.shape
            if type(X)!= np.array:
                X=np.array(x)[0]
            return X
        except :
            cov = np.array(X)
            try:
                len(cov)
                return cov
            except :
                return np.eye(n) * X




    def gaussian(X,mean,var):
        """
        compute the normal distribution of x with the mean mean
         and the varirance var
        """

        return (np.exp((-0.5*(np.asarray(X)-mean)**2)/var)/ math.sqrt(2*math.pi*var))


    def mutivariate_gaussian(X,mean,cov):
        X= np.array(X,copy=False,ndmin=1).flatten()
        mean= np.array(mean,copy=False,ndmin=1).flatten()

        nx = len(mean)
        cov = to_cov(cov, nx)

        norm_coef = nx*math.log(2*math.pi)  + np.linalg.slogdet(cov)[1]

        error = X - mean

        if(sp.issparse(cov)):
            numerator= spln.spsolve(cov , error).T.dot(error)
        else:
            numerator  = np.linalg.solve(cov ,error).T.dot(error)

        return math.exp(-0.5*(norm_coef + numerator))





data = read("hw6_data.txt")
x,y,theta = get_position(data)

#L,R = buildmap(data,x,y,theta)
plt.plot(x,y,c='r')

#x = [data[0] for data in L]
#y = [data[1] for data in L]
#plt.scatter(x,y,c='b',marker='.')


#x = [data[0] for data in R]
#y = [data[1] for data in R]
#plt.scatter(x,y,c='g',marker='.')

plt.legend()
plt.show()
