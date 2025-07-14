#######################################################################
# Its about to get wicked up in here:
# Molecular Dynamics based control policy
# 
#
# Author: Humzah Durrani
# STATUS: In-Progress
# To do: Phase Change from Flocking (Liquid) to Solid (Crystallization)
# 
# Sources/References:
#
#######################################################################


##########################
# Importing Libraries
##########################
import numpy as np
import matplotlib.pyplot as plt
import lj_swarm

##########################
# Parameters
##########################

#Number of Agents
agents = 100

#Max Steps (Animation run-time)
max_steps = 1000
counter = 1

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
gamma_pos1 = [10,10]
gamma_pos2 = [40, 40]

#Temperature Control
temp = 4

##########################
# Main Loop
##########################

#Initialize Agents
multi_agent_sys = lj_swarm.multi_agent(agents)
plt.figure()
ax  = plt.gca()

for steps in range(max_steps):
    
    plt.cla()
    plt.title(f't = {steps}')
    ax.set_xlim(0,50)
    ax.set_ylim(0,50)

    if counter >= 300: gamma_pos = gamma_pos2
    else: gamma_pos = gamma_pos1

    forces = multi_agent_sys.compute_forces(cutoff, epsilon, sigma, gamma_pos, c1_gamma, c2_gamma, temp, counter)
    multi_agent_sys.update(forces, max_speed)

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
        
    xg, yg = gamma_pos
    ax.scatter(xg, yg, s=5, c='blue', marker = '*')
    
    counter += 1
    plt.pause(0.01)
    
plt.show()