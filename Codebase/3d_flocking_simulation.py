#######################################################################
# This 3-D Flocking Simulation is based off of the original
# Craig Reynolds Paper and Algorithm. There are a ton like it.
# However, this one is 3-D which makes it alittle cooler to see. 
# Enjoy!
#######################################################################

##########################
#Importing Libraries
##########################
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

##########################
#Parameters (Please Tune!)
##########################

#Number of Agents
boid_count = 100

#Random Starting Positon range (X,Y,Z)
lower_pos = np.array([100,900,200])
upper_pos = np.array([200,1100,800])


#Velocity Range (X,Y,Z)
lower_vel = np.array([0,-2,-1])
upper_vel = np.array([1,2,1])

#Force Applied to want to move to the center:
middle_strength = 0.01

#Separation: Alert Distance
alert_dist = 10

#Match Speed 
flying_dist = 10
flying_str = 0.0125

##########################
# Simulation
##########################
sim_limits = np.array([2000,2000,2000])

#Function: Flock Set-up used for both position and velocity
def flock(boid_count, lower_lim, upper_lim):
    width = upper_lim - lower_lim
    return lower_lim[:, np.newaxis] + np.random.rand(3,boid_count) * width[:, np.newaxis]

#Function: Bounding Walls


#Function: Update Boids Position
def update_sim(positions, velocities, middle_strength, alert_dist, flying_dist, flying_str):
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
    velocities += np.sum(sep_if_close,1)

    #Match Speed Update
    vel_diff = velocities[:, np.newaxis, :] - velocities[:, :, np.newaxis]
    very_far = square_dist > flying_dist
    vel_diff_if_close = np.copy(vel_diff)
    vel_diff_if_close[0,:,:][very_far] = 0
    vel_diff_if_close[1,:,:][very_far] = 0
    velocities -= np.mean(vel_diff_if_close, 1) * flying_str

    #Position Update
    positions += velocities

    #Keeps the agents in bound
    positions = np.mod(positions, sim_limits[:, np.newaxis])

    return positions, velocities


#Function: Animate the simulation
def animate(frame):
    global positions, velocities
    positions, velocities = update_sim(positions, velocities, middle_strength, alert_dist, flying_dist, flying_str)
    scatter._offsets3d = (positions[0], positions[1], positions [2])

##########################
# Run Code
##########################

#Position Setup
positions = flock(boid_count,lower_pos,upper_pos)

#Velocity Setup
velocities = flock(boid_count,lower_vel,upper_vel)


#Display Figure
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
scatter = axes.scatter(positions[0,:], positions[1,:], positions[2,:], marker="P", edgecolor="k", lw=0.5)
scatter
axes.set_xlim(0, sim_limits[0])
axes.set_ylim(0, sim_limits[1])
axes.set_zlim(0, sim_limits[2])

anime = animation.FuncAnimation(figure, animate, frames=50, interval=50, blit=False)
plt.show()
