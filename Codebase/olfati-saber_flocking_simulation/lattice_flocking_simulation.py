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
#  4. ChatGPT
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

#Number of Agents
agents = 30

#Max Steps (Animation run-time)
max_steps = 1000

#Smoothing factor for sigma_norm
eps = 0.1

#Shapes phi function
A = 5
B = 5

#Normalization Constant
C = np.abs(A - B) / np.sqrt(4 * A * B)

#Bump Function Threshold
H = 0.2

#Range and Distance
R = 12
D = 10


#Control Gain Parameters from the paper
#Inter-agent interactions
c1_alpha = 3
c2_alpha = 2 * np.sqrt(c1_alpha)
# Global Control
c1_gamma = 5
c2_gamma = 0.2 * np.sqrt(c1_gamma)

##########################
# Core Math Functions
##########################

#Smooths transition between 1 and 0 when z crosses threshold
def bump_funct(z):
    Ph = np.zeros_like(z)
    Ph[z <= 1] = (1+ np.cos(np.pi * (z[z <= 1] - H) / (1 - H))) / 2
    Ph[z < H] = 1
    Ph[z < 0] = 0
    return Ph

#Smooths version of identity function that avoids singularities at 0
def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)

#Used to compute distances with smooth behavior
def sigma_norm(z):
    return (np.sqrt(1 + eps * np.linalg.norm(z, axis =-1, keepdims=True) ** 2) - 1) / eps

#Gradient of sigma norm, to compute normalized direction vectors 
#for force field
def sigma_grad(z):
    return z / np.sqrt(1 + eps * np.linalg.norm(z, axis =-1, keepdims=True) ** 2)

#Base Potential Function
def phi(z):
    return ((A + B) * sigma_1(z + C) + (A - B)) / 2

#Function that uses bump function to restrict the range of interaction
def phi_alpha(z):
    r_alpha = sigma_norm([R])
    d_alpha = sigma_norm([D])
    return bump_funct(z / r_alpha) * phi(z - d_alpha)


##########################
# Multi-Agent Class
##########################
class multi_agent:
    
    #Initialize agents position and velocity
    def __init__(self, number, sampletime=0.1):
        self.dt = sampletime
        self.agents = np.random.randint(0, 100, (number,2)).astype('float')
        self.agents = np.hstack([self.agents, np.zeros((number,2))])

    #Update agents position and velocity
    def update(self,u=2):
        q_dot = u
        self.agents[:,2:] += q_dot * self.dt
        p_dot = self.agents[:,2:]
        self.agents[:,:2] += p_dot * self.dt

##########################
# Helper Functions
##########################

#Function that indicates whether two agents are within interaction range
def get_adj_mat(nodes, r):
    n = len(nodes)
    adj = np.zeros((n,n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(nodes[i,:2] - nodes[j, :2])
                adj[i,j] = dist <= r
    return adj

#Function for influence coeffs based on distance (velocity matching)
def influence(q_i, q_js):
    r_alpha = sigma_norm([R])
    return bump_funct(sigma_norm(q_js - q_i) / r_alpha)

#Function to calculate direction vectors (agents to neighbors)
def local_dir(q_i, q_js):
    return sigma_grad(q_js - q_i)

##########################
# Main Loop
##########################

#Initialize Agents
multi_agent_sys = multi_agent(number = agents)

for i in range(max_steps):
    #Compute Adjacency
    adj_mat = get_adj_mat(multi_agent_sys.agents, R)
    u = np.zeros((agents, 2))

    #Loop through agents
    for j in range(agents):
        #Get positions and velocity
        agent_p = multi_agent_sys.agents[j, :2]
        agent_q = multi_agent_sys.agents[j, 2:]

        #Init control input
        u_alpha = np.zeros(2)

        #Identify and process neighbors
        neighbor_idx = adj_mat[j]
        if np.sum(neighbor_idx) > 1:
            neighbor_p = multi_agent_sys.agents[neighbor_idx, :2]
            neighbor_q = multi_agent_sys.agents[neighbor_idx, 2:]
            direction = local_dir(agent_p, neighbor_p)

            #Interaction with neighbors
            u1 = c2_alpha * np.sum(phi_alpha(sigma_norm(neighbor_p - agent_p)) * direction, axis=0)

            #Velocity alignment with neighbors
            n_influence = influence(agent_p, neighbor_p)
            u2 = c2_alpha * np.sum(n_influence * (neighbor_q - agent_q), axis=0)
        
            #Total Influence
            u_alpha = u1 + u2 

        #Feedback from gamma agent
        u_gamma = -c1_gamma * sigma_1(agent_p - [50,50]) - c2_gamma * agent_q

        #Total control input
        u[j] = u_alpha + u_gamma

    #Update agent states
    multi_agent_sys.update(u)

    #Plot Agents and their connections
    plt.cla()
    plt.axis([0,100,0,100])

    for k in range(agents):
        for l in range(agents):
            if k != l and adj_mat[k,l] == 1:
                plt.plot(multi_agent_sys.agents[[k,l], 0], multi_agent_sys.agents[[k,l],1])

    for m, (x,y, _, _) in enumerate(multi_agent_sys.agents):
        plt.scatter(x,y, c='black')
    
    plt.pause(0.01)

plt.show()