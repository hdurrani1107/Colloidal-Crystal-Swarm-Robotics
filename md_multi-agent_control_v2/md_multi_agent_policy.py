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
        self.agent_states = [{
            "temp": 10.0,
            "sigma": 30.0,
            "crystallized": False,
            "stuck_counter": 0,
            "last_pos": np.zeros(2)
            } for _ in range(number)]
    
    def compute_forces(self, cutoff, epsilon, sigma, gamma_pos, c1_gamma, c2_gamma, obstacles, repulsion_strength, apply_gamma, target_goals, agent_sigmas):
        n = len(self.agents)
        forces = np.zeros((n,2))
        SIM_BOUNDS = [0, 600]
        PIXELS_PER_UNIT = 600 / (SIM_BOUNDS[1] - SIM_BOUNDS[0])
        for i in range (n):
            pos_i = self.agents[i, :2]
            vel_i = self.agents[i, 2:]
            sigma_i = agent_sigmas[i]
            total_force = np.zeros(2)
            #objective = pos_i - gamma_pos

            #obstacle repulsion
            for obs in obstacles:
                obs_pos = obs["pos"] / PIXELS_PER_UNIT  # convert to sim units
                obs_radius = obs["radius"] / PIXELS_PER_UNIT
                offset = pos_i - obs_pos
                dist = np.linalg.norm(offset)
                if dist < obs_radius + 2.0:  # buffer
                    direction = offset / (dist + 1e-6)
                    penetration = max(0,0, obs_radius + 2.0 - dist)
                    repulse = repulsion_strength * (penetration ** 2)
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
                    inv_r6 = (sigma_i * inv_r) ** 6
                    inv_r12 = inv_r6 ** 2
                    lj_scalar = 24* (sigma_i) * (2 * inv_r12 - inv_r6) * inv_r
                    total_force += lj_scalar * (offset / dist) 
            
            if apply_gamma[i] and target_goals[i] is not None:
                # Get closest goal
                goal_pos = target_goals[i]
                objective = goal_pos - pos_i
                u_gamma = c1_gamma * sigma_1(objective) - c2_gamma * vel_i
                total_force += u_gamma

            forces[i] = total_force

        return forces
    
    def update(self, forces, max_speed, c1_lang, c2_lang, mass, c3_lang):

        for i in range(len(self.agents)):
            v = self.agents[i, 2:]
            f = forces[i]

            # Generate Gaussian noise
            noise = c3_lang * np.random.normal(0,1, size=2)

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
            elif x > 700:
                self.agents[i, 0] = 700
                self.agents[i, 2] *= -1

            if y < 0:
                self.agents[i, 1] = 0
                self.agents[i, 3] *= -1
            elif y > 700:
                self.agents[i, 1] = 700
                self.agents[i, 3] *= -1 