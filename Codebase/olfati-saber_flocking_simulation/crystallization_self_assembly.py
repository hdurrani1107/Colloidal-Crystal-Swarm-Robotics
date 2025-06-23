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


##########################
# Parameters
##########################

#Number of Agents
agents = 20

#Max Steps (Animation run-time)
max_steps = 1000


#Repulsion Agents
obstacles = np.array([[10,10], [40,40], [10, 40], [30, 10]])
R_obs = 5

#Agent-Radius, Interaction Radius, Max-Speed
agent_radius = 2
interact_radius = 5
max_speed = 2

#Lennard-Jones Interaction Coefficients
epsilon = 1.0
sigma = 3.0
cutoff = 3 * sigma
optimal_range = (0.9 * sigma, 1.1 * sigma)

#Gamme Control
c1_gamma = 5
c2_gamma = 0.2 * np.sqrt(c1_gamma)

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
        def generate_valid_positions(n_agents, obstacles, R_obs, bounds=(0,50)):
            valid_positions = []
            while len(valid_positions) < n_agents:
                candidate = np.random.uniform(bounds[0], bounds[1], 2)
                if is_outside_obstacles(candidate, obstacles, R_obs):
                    valid_positions.append(candidate)
            return np.array(valid_positions)
        
        #Generate valid positions
        positions = generate_valid_positions(number, obstacles, R_obs)
        self.agents = np.hstack([positions, np.zeros((number,2))])

    def compute_forces(self, obstacles, R_obs, interact_radius):
        n = len(self.agents)
        forces = np.zeros((n,2))
        for i in range (n):
            pos_i = self.agents[i, :2]
            force = np.zeros(2)

            #Agent-Agent Repulsion
            for j in range(n):
                if i != j:
                    pos_j = self.agents[j, :2]
                    offset = pos_i - pos_j
                    dist = np.linalg.norm(offset)
                    if dist < cutoff and dist > 1e-3:
                        r_unit = offset/dist
                        lj_scalar = 24* epsilon * ((2 * (sigma**12) / dist**13) - ((sigma**6) / dist**7))
                        force += lj_scalar * r_unit
            
            for obs in obstacles:
                offset = pos_i - obs
                dist = np.linalg.norm(offset)
                if dist < R_obs:
                    force += offset / (dist**2 + 1e-3)

            forces[i] = force

        return forces
    
    def update(self, forces, noise_scale):
        
        #Brownian Motion
        noise = np.random.normal(0, noise_scale, size = self.agents[:, 2:].shape)

        self.agents[:, 2:] += (forces+noise) * self.dt
        speeds = np.linalg.norm(self.agents[:, 2:], axis=1)
        too_fast = speeds > max_speed
        self.agents[too_fast, 2:] *= (max_speed / speeds[too_fast])[:, None]

        #Update Positions
        self.agents[:, :2] += self.agents[:,2:] * self.dt


##########################
# Main Loop
##########################

#Initialize Agents
multi_agent_sys = multi_agent(number = agents)
plt.figure()
ax  = plt.gca()
ax.set_xlim(0,50)
ax.set_ylim(0,50)


for steps in range(max_steps):
    
    plt.cla()
    plt.title(f't = {steps}')
    forces = multi_agent_sys.compute_forces(obstacles, R_obs, interact_radius)
    multi_agent_sys.update(forces, noise_scale=0.3)

    for obs in obstacles:
        circle = plt.Circle(obs, R_obs, color = 'red', fill = 'True', linestyle = '-')
        ax.add_patch(circle)
        plt.scatter(*obs, c='red', marker='x')

    for m, (x,y, _, _) in enumerate(multi_agent_sys.agents):
        plt.scatter(x,y, s=1, c='green', marker = 's', alpha=0.6)

    for i in range(agents):
        for j in range(i + 1, agents):
            pos_i = multi_agent_sys.agents[i, :2]
            pos_j = multi_agent_sys.agents[j, :2]
            offset = pos_i - pos_j
            dist = np.linalg.norm(offset)

            if dist < cutoff:
                if optimal_range[0] <= dist <= optimal_range[1]:
                    color = 'green'
                elif dist < optimal_range[0] or dist > optimal_range[1]:
                    color = 'yellow' if dist < 1.5 * sigma else 'red'
                
                plt.plot([pos_i[0], pos_j[0]],[pos_i[1], pos_j[1]], color = color, linewidth=0.5)

    plt.pause(0.01)
    
#plt.title('t = {steps}')
plt.show()