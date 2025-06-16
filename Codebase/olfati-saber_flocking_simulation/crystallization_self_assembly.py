#######################################################################
# Its about to get wicked up in here:
# Crystallization Self-Assembly Algorithm (See Sources)
# 
#
# Author: Humzah Durrani
# STATUS: In-Progress
# To do: Boundary Configuration, Temperature Dynamic, Obstacles?, 3-D?   
# 
# Sources/References:
#
#######################################################################

##########################
# Importing Libraries
##########################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


##########################
# Parameters
##########################

#Number of Agents
agents = 100

#Max Steps (Animation run-time)
max_steps = 1000


#Repulsion Agents
obstacles = np.array([[25,25,25], [75,50,50], [10, 90, 90]])
R_obs = 5


##########################
# Multi-Agent Class
##########################
class multi_agent:
    
    #Initialize agents position and velocity
    def __init__(self, number, sampletime=0.1):
        self.dt = sampletime

        #Calculates if agent is outside obstacle at start
        def is_outside_obstacles(pos, obstacles, R_obs):
            return all(np.linalg.norm(pos-obs) > R_obs for obs in obstacles)
        
        #Ensures agent is in a valid position outside obstacles at start
        def generate_valid_positions(n_agents, obstacles, R_obs, bounds=(0,100)):
            valid_positions = []
            while len(valid_positions) < n_agents:
                candidate = np.random.uniform(bounds[0], bounds[1], 3)
                if is_outside_obstacles(candidate, obstacles, R_obs):
                    valid_positions.append(candidate)
            return np.array(valid_positions)
        
        #Generate valid positions
        positions = generate_valid_positions(number, obstacles, R_obs)
        self.agents = np.hstack([positions, np.zeros((number,3))])



##########################
# Main Loop
##########################

#Initialize Agents
multi_agent_sys = multi_agent(number = agents)

#for i in range(max_steps):
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

def plot_sphere(ax, center, radius, color='red', alpha=0.2):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

for obs in obstacles:
    plot_sphere(ax, obs, R_obs)

for m, agents in enumerate(multi_agent_sys.agents):
    x,y,z = agents[:3]
    ax.scatter(x,y,z, s=5, c='green', marker = 's')

#for obs in obstacles:
#    circle = plt.Circle(obs, R_obs, color = 'red', fill = 'True', linestyle = '-')
#    plt.gca().add_patch(circle)
#    plt.scatter(*obs, c='red', marker='x')

#for m, (x,y, _, _) in enumerate(multi_agent_sys.agents):
#    plt.scatter(x,y, s=1, c='green', marker = 's')

#plt.pause(0.01)
plt.show()