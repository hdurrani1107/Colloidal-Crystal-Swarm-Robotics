#######################################################################
# Its about to get wicked up in here:
# Molecular Dynamics based control policy
# 
#
# Author: Humzah Durrani
# STATUS: In-Progress
# To do: Holding off on 3-D, Phase Change Dynamic (See phase_change_simulation)   
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

obstacle1 = np.array([[20,0],[30,0],[30,22],[20,22],[20,0]])
obstacle2 = np.array([[20,30],[30,30],[30,50],[20,50],[20,30]])
obstacles = [obstacle1, obstacle2]
R_obs = 10

#Temperature Control
temp = 2


##########################
# Helper Function
##########################

#Prevent Singularities for Gamma Function
def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)

def closest_obs_edge(agent,obstacle):
    closest_point = None
    min_dist = np.inf
    for edge in range(len(obstacle)-1):
        a = obstacle[edge]
        b = obstacle[edge + 1]

        ab = b - a
        ap = agent - a
        ab_squared = np.dot (ab, ab)
        if ab_squared == 0: proj = a
        else:
            t = np.dot(ap, ab) / ab_squared
            t = np.clip(t, 0 ,1)
            proj = a + t * ab

        dist = np.linalg.norm(agent - proj)
        if dist < min_dist:
            min_dist = dist
            closest_point = proj

    return closest_point, min_dist


##########################
# Multi-Agent Class
##########################
class multi_agent:
    
    #Initialize agents position and velocity
    def __init__(self, number, sampletime=0.1, obstacles = None):
        self.dt = sampletime
        self.obstacles = obstacles if obstacles is not None else []

        #Calculates if agent is outside obstacle at start
        def is_outside_obstacles(pos, obstacles, R_obs):
            return all(np.linalg.norm(pos-obs) > R_obs for obs in obstacles)
        
        #Ensures agent is in a valid position outside obstacles at start
        def generate_valid_positions(n_agents, obstacles, R_obs, bounds=(0,10)):
            valid_positions = []
            while len(valid_positions) < n_agents:
                candidate = np.random.uniform(bounds[0], bounds[1], 2)
                if is_outside_obstacles(candidate, obstacles, R_obs):
                    valid_positions.append(candidate)
            return np.array(valid_positions)
        #Generate valid positions
        #for i in range(number):
        #    positions = np.zeros((number, 2))
        #    candidate = np.random.uniform(0,10,2)
        #    positions = np.vstack((candidate, i))
        #    print(positions)
            
        positions = generate_valid_positions(number, obstacles, R_obs) 
        #print(np.array(positions))
        self.agents = np.hstack([np.array(positions), np.zeros((number,2))])
    
    def compute_forces(self):
        n = len(self.agents)
        forces = np.zeros((n,2))
        for i in range (n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            total_force = np.zeros(2)
            gamma_pos = [40, 40]
            if counter >= 300:
                gamma_pos = [10,10]
            objective = pos_i - gamma_pos

            #Temperature Control
            t_control = np.linalg.norm(objective)
            temp_decay = ((1.01) ** (t_control - 5)) + (1e-3)
            #temp_decay = 1

            #LEONARD JONES POTENTIAL
            for j in range(n):
                if i != j:
                    pos_j = self.agents[j, :2]
                    offset = pos_i - pos_j
                    dist = np.linalg.norm(offset)
                    if dist < cutoff and dist > 1e-3:
                        r_unit = offset/dist
                        lj_scalar = 24* (epsilon / temp_decay) * ((2 * ((sigma * temp_decay)**12) / dist**13) - (((sigma * temp_decay)**6) / dist**7))
                        total_force += lj_scalar * r_unit 
            

            gamma_pos = [40, 40]
            if counter >= 300:
                gamma_pos = [10,10]
            objective = pos_i - gamma_pos
            u_gamma = -c1_gamma * sigma_1(objective) - c2_gamma * vel_i
            total_force += u_gamma

            for obs in self.obstacles:
                closest_point, min_dist = closest_obs_edge(pos_i, obs)

                if min_dist < R_obs:
                    direction = pos_i - closest_point
                    norm = np.linalg.norm(direction)

                    if norm != 0:
                        direction_unit = direction / norm
                        magnitude = 20 * (1 / min_dist - 1 / R_obs) / (min_dist ** 2)
                        rep_u = magnitude * direction_unit
                        total_force += rep_u


            forces[i] = total_force

        return forces
    
    def update(self, forces):

        #Brownian Motion Removed because its strange
        #noise = np.random.normal(0, 1, size = self.agents[:, 2:].shape)

        self.agents[:, 2:] += (forces) * self.dt
        speeds = np.linalg.norm(self.agents[:, 2:], axis=1)
        too_fast = speeds > max_speed
        self.agents[too_fast, 2:] *= (max_speed / speeds[too_fast])[:, None]

        #Update Positions
        self.agents[:, :2] += self.agents[:,2:] * self.dt

         # --- Boundary Check and Bounce ---
        # Bounds
        x_min, x_max = 0, 50
        y_min, y_max = 0, 50

        for i in range(len(self.agents)):
            x, y = self.agents[i, :2]
            vx, vy = self.agents[i, 2:]

            # Check X boundaries
            if x < x_min:
                self.agents[i, 0] = x_min
                self.agents[i, 2] *= -1  # Reflect x-velocity
            elif x > x_max:
                self.agents[i, 0] = x_max
                self.agents[i, 2] *= -1

            # Check Y boundaries
            if y < y_min:
                self.agents[i, 1] = y_min
                self.agents[i, 3] *= -1  # Reflect y-velocity
            elif y > y_max:
                self.agents[i, 1] = y_max
                self.agents[i, 3] *= -1


##########################
# Main Loop
##########################

#Initialize Agents
multi_agent_sys = multi_agent(agents, obstacles=obstacles)
plt.figure()
ax  = plt.gca()
rectangle1 = plt.Rectangle((20, 0), 10, 22, facecolor='red', alpha=0.7)
rectangle2 = plt.Rectangle((20, 30), 10, 20, facecolor='red', alpha=0.7)


for steps in range(max_steps):
    
    plt.cla()
    plt.title(f't = {steps}')
    ax.set_xlim(0,50)
    ax.set_ylim(0,50)
    forces = multi_agent_sys.compute_forces()
    multi_agent_sys.update(forces)
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)

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
    
    counter += 1
    plt.pause(0.01)
    
plt.show()