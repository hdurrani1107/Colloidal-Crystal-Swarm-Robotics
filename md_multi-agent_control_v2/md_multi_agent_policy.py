#######################################################################
# md_multi_agent_policy.py
#
# MD Engine multi agent control policy
#
# Author: Humzah Durrani
#######################################################################


##########################
# Importing Libraries
##########################
import numpy as np

##########################
# Helper Function
##########################
def sigma_1(z):
    return z / np.sqrt(1 + z ** 2)


##########################
# Multi-Agent Class
##########################
class multi_agent:
    
    #Initialize agents position and velocity
    def __init__(self, number, sampletime=0.1, bounds=[0,50]):
        self.dt = sampletime
        positions = []
        while len(positions) < number:
            candidate = np.random.uniform(bounds[0], bounds[1], 2)
            positions.append(candidate)
        positions = np.array(positions)
        self.agents = np.hstack([np.array(positions), np.zeros((number,2))])
    
    def compute_forces(self, cutoff, epsilon, sigma, gamma_pos, c1_gamma, c2_gamma, obstacles, repulsion_strength=100.0):
        n = len(self.agents)
        forces = np.zeros((n,2))
        SIM_BOUNDS = [0, 600]
        PIXELS_PER_UNIT = 600 / (SIM_BOUNDS[1] - SIM_BOUNDS[0])
        for i in range (n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            total_force = np.zeros(2)
            objective = pos_i - gamma_pos

            #obstacle repulsion
            for obs in obstacles:
                obs_pos = obs["pos"] / PIXELS_PER_UNIT  # convert to sim units
                obs_radius = obs["radius"] / PIXELS_PER_UNIT
                offset = pos_i - obs_pos
                dist = np.linalg.norm(offset)
                if dist < obs_radius + 1.0:  # buffer
                    direction = offset / (dist + 1e-6)
                    repulse = repulsion_strength * (1.0 / (dist + 1e-6) - 1.0 / obs_radius)
                    total_force += repulse * direction


            #LJ POTENTIAL
            for j in range(n):
                if i == j:
                    continue
                pos_j = self.agents[j, :2]
                offset = pos_i - pos_j
                dist = np.linalg.norm(offset)
                if dist < cutoff and dist > 1e-3:
                    inv_r = 1.0/dist
                    inv_r6 = (sigma * inv_r) ** 6
                    inv_r12 = inv_r6 ** 2
                    lj_scalar = 24* (epsilon) * (2 * inv_r12 - inv_r6) * inv_r
                    total_force += lj_scalar * (offset / dist) 
            
            if gamma_pos:
                # Get closest goal
                goal_distances = [np.linalg.norm(pos_i - (g / PIXELS_PER_UNIT)) for g in gamma_pos]
                closest_goal = gamma_pos[np.argmin(goal_distances)]
                objective = closest_goal / PIXELS_PER_UNIT - pos_i
                u_gamma = c1_gamma * sigma_1(objective) - c2_gamma * vel_i
                total_force += u_gamma

            forces[i] = total_force

        return forces
    
    def update(self, forces, max_speed, c1_lang, c2_lang, mass):

        for i in range(len(self.agents)):
            v = self.agents[i, 2:]
            f = forces[i]

            # Generate Gaussian noise
            noise = np.random.normal(0, c2_lang[i], size=2)

            # Langevin velocity update
            v_new = (
                v * c1_lang[i] +                         # friction decay
                (f / mass) * self.dt +                   # deterministic force
                c2_lang[i] * noise                       # stochastic force
            )

            # Clamp speed to max
            speed = np.linalg.norm(v_new)
            if speed > max_speed:
                v_new = (v_new / speed) * max_speed

            self.agents[i, 2:] = v_new
            self.agents[i, :2] += v_new * self.dt  # Position update

            # --- Boundary Check and Bounce ---
            x, y = self.agents[i, :2]

            if x < 0:
                self.agents[i, 0] = 0
                self.agents[i, 2] *= -1
            elif x > 600:
                self.agents[i, 0] = 600
                self.agents[i, 2] *= -1

            if y < 0:
                self.agents[i, 1] = 0
                self.agents[i, 3] *= -1
            elif y > 600:
                self.agents[i, 1] = 600
                self.agents[i, 3] *= -1