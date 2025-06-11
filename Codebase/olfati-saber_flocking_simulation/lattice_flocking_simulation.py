#######################################################################
# This is an updated flocking simulation based on:
# Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory
# by Reza Olfati-Saber
#
# Author: Humzah Durrani
# STATUS: In-Progress
# Sources/References:
#  1. R. Olfati-Saber. Flocking for multi-agent dynamic systems: 
#     algorithms and theory. IEEE Transactions on Automatic Control, 
#     51(3):401–420, 2006.
#  2. https://github.com/arbit3rr/Flocking-Multi-Agent/tree/main
#  3. https://github.com/tjards/flocking_network?tab=readme-ov-file
#
# 
#######################################################################

##########################
# Importing Libraries
##########################
import numpy as np
import math
import matplotlib.pyplot as plt

##########################
# Parameters
##########################
#Smoothing factor for sigma_norm
eps = 0.1

#Shapes phi function
A = 5
B = 5

#Normalization Constant
C = np.abs(A - B) / np.sqrt(4 * A * B)

#Bump Function Threshold
H = 0.2

##########################
# Core Math Functions
##########################

#Smooths transition between 1 and 0 when z crosses threshold
def bump_funct(z):
    Ph = np.zeros_like(z)
    Ph[z <= 1] = (1+ np.cos(np.pi * (z[z <= 1] - H) / (1 - H))) / 2
    Ph[z < H] = 1
    Ph[z < 0] = 0

#Smooths version of identity function that avoids singularities at 0
def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)

#Used to compute distances with smooth behavior
def sigma_norm(z):
    return (np.sqrt(1 + eps * np.linalg.norm(z, axis =-1, keepdims=True) ** 2) - 1) / eps

#Gradient of sigma norm, to compute normalized direction vectors 
#for force field
def sigma_grad(z):
    return z / np.sqrt(1 + eps * np.linalg.norm(z axis =-1, keepdims=True) ** 2)

#Base Potential Function
def phi(z):
    return ((A + B) * sigma_1(z + C) + (A - B)) / 2

#Function that uses bump function to restrict the range of interaction
def phi_alpha(z):
    r_alpha = sigma_norm([R])
    d_alpha = sigma_norm([D])
    return bump_function(z / r_alpha) * phi(z - d alpha)


##########################
# Multi-Agent Class
##########################
class multi_agent:
    def __init__(self, number, sampletime=0.1):
        self.dt = sampletime
        self.agents = np.randomint(0, 100, (number,2)).astype('float')
        self.agents = hstack([self.agents, np.zeros((number,2))])

    def update(self,u=2):
        q_dot = u
        self.agents[:,2:] += q_dot * self.dt
        p_dot = self.agents[:,2:]
        self.agents[:,:2] += p_dot * self.dt

##########################
# Helper Functions
##########################

