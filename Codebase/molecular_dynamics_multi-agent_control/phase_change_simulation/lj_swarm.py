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
    
    def compute_forces(self, cutoff, epsilon, sigma, gamma_pos, c1_gamma, c2_gamma, temp, counter):
        n = len(self.agents)
        forces = np.zeros((n,2))
        for i in range (n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            total_force = np.zeros(2)
            objective = pos_i - gamma_pos

            #Temperature Control
            t_control = np.linalg.norm(objective)
            temp_decay = (temp - (counter * 0.1)) + 1e-3
            if temp_decay < 1: temp_decay = 1
            #print(temp_decay)
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
            
            objective = pos_i - gamma_pos
            u_gamma = -c1_gamma * sigma_1(objective) - c2_gamma * vel_i
            total_force += u_gamma

            forces[i] = total_force

        return forces
    
    def update_sim(positions, velocities, middle_strength, alert_dist, flying_dist, flying_str, margin, turn_factor):
        #Move to middle
        middle = np.mean(positions,1)
        middle_direction = positions - middle[:,np.newaxis]
        
        #Velocity Update
        velocities -= middle_direction * middle_strength

        #Separation Update
        separations = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]
        squared_disp = separations * separations
        square_dist = np.sum(squared_disp,  0)
        far_away = square_dist > alert_dist
        sep_if_close = np.copy(separations)
        sep_if_close[0,:,:][far_away] = 0
        sep_if_close[1,:,:][far_away] = 0
        sep_if_close[2,:,:][far_away] = 0
        velocities += np.sum(sep_if_close,1)

        #Match Speed Update
        vel_diff = velocities[:, np.newaxis, :] - velocities[:, :, np.newaxis]
        very_far = square_dist > flying_dist
        vel_diff_if_close = np.copy(vel_diff)
        vel_diff_if_close[0,:,:][very_far] = 0
        vel_diff_if_close[1,:,:][very_far] = 0
        vel_diff_if_close[2,:,:][very_far] = 0
        velocities -= np.mean(vel_diff_if_close, 1) * flying_str

        #Position Update
        positions += velocities

        #Keeps the agents in bound
        #positions = np.mod(positions, sim_limits[:, np.newaxis])
        for axis in range(3):
            dist_to_low = positions[axis] - 0
            dist_to_high = sim_limits[axis] - positions[axis]

            velocities[axis] += (dist_to_low < margin) * (turn_factor * (margin - dist_to_low) / margin)
            velocities[axis] -= (dist_to_high < margin) * (turn_factor * (margin - dist_to_high) / margin)

        return positions, velocities
    
    def update(self, forces, max_speed):

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
# Multi-Agent Class
##########################