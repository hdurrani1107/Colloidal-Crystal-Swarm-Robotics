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
    def __init__(self, number, temp, sampletime=0.1, bounds=[0,50]):
        self.dt = sampletime
        self.kB = 2
        self.mass = 1
        self.temp = temp
        positions = []
        while len(positions) < number:
            candidate = np.random.uniform(bounds[0], bounds[1], 2)
            positions.append(candidate)
        positions = np.array(positions)

        #Maxwell-Boltzman Distribution for initial State
        #Look to add to force update
        sigma_v = np.sqrt(self.kB * temp / self.mass)  # m/s
        velocities = np.random.normal(0, sigma_v, size=(number, 2))

        # Remove center-of-mass drift
        velocities -= np.mean(velocities, axis=0)

        # Rescale to exact target temperature
        KE = 0.5 * self.mass * np.sum(velocities**2)  # total kinetic energy
        T_inst = (2 * KE) / (2 * number * self.kB)
        lambda_rescale = np.sqrt(temp / T_inst)
        velocities *= lambda_rescale

        self.agents = np.hstack([np.array(positions), np.array(velocities)])
    
    def compute_forces(self, cutoff, epsilon, sigma, gamma_pos, c1_gamma, c2_gamma):
        n = len(self.agents)
        forces = np.zeros((n,2))
        for i in range (n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            total_force = np.zeros(2)
            objective = pos_i - gamma_pos

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
            
            objective = gamma_pos - pos_i
            u_gamma = c1_gamma * sigma_1(objective) - c2_gamma * vel_i
            total_force += u_gamma

            forces[i] = total_force

        return forces
    
    def update(self, forces, max_speed, c1_lang, c2_lang, mass):

        #Noise
        noise = np.random.normal(0, 1, size = self.agents[:, 2:].shape)

        #Langevin Velocity Verlet
        self.agents[:, 2:] = (((forces / mass) * self.dt) + (c2_lang * noise) + (self.agents[:, 2:] * c1_lang))
        speeds = np.linalg.norm(self.agents[:, 2:], axis=1)
        too_fast = speeds > max_speed
        self.agents[too_fast, 2:] *= (max_speed / speeds[too_fast])[:, None]

        #Update Positions
        self.agents[:, :2] += self.agents[:,2:] * self.dt

         # --- Boundary Check and Bounce ---
        # Bounds
        x_min, x_max = 0, 600
        y_min, y_max = 0, 600

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