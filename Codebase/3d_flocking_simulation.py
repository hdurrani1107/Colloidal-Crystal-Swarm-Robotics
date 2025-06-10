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
x_position = np.array([100,900])
y_position = np.array([200,1100])
#z_position = np.array([200,900])

#Velocity Range (X,Y,Z)
x_vel = np.array([0,-20])
y_vel = np.array([10,20])
#z_vel = np.array([-10,10])

#Force Applied to want to move to the center:
middle_strength = 0.01

#Separation: Alert Distance
alert_dist = 100

#Match Speed 
flying_dist = 1000
flying_str = 0.125

##########################
# Simulation
##########################
sim_limits = np.array([2000,2000])

#Function: Random starting positions for boid
def flock(boid_count, lower_lim, upper_lim):
    width = upper_lim - lower_lim
    return lower_lim[:, np.newaxis] + np.random.rand(2,boid_count) * width[:, np.newaxis]

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


#Function: Animate the simulation
def animate(frame):
    update_sim(positions, velocities, middle_strength, alert_dist, flying_dist, flying_str)
    scatter.set_offsets(positions.transpose())

##########################
# Run Code
##########################

#Position Setup
positions = flock(boid_count,x_position,y_position)

#Velocity Setup
velocities = flock(boid_count,x_vel,y_vel)


#Display Figure
figure = plt.figure()
axes = plt.axes(xlim=(0,sim_limits[0]), ylim=(0,sim_limits[1]))
scatter = axes.scatter(positions[0,:], positions[1,:], marker="v", edgecolor="k", lw=0.5)
scatter

anime = animation.FuncAnimation(figure, animate, frames=50, interval=50)
plt.show()
