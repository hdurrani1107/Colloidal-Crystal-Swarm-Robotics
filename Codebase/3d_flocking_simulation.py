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

##########################
# Simulation
##########################
sim_limits = np.array([2000,2000])

#Function: Random starting positions for boid
def flock(boid_count, lower_lim, upper_lim):
    width = upper_lim - lower_lim
    return lower_lim[:, np.newaxis] + np.random.rand(2,boid_count) * width[:, np.newaxis]

#Function: Update Boids Position
def update_sim(positions, velocities, middle_strength):
    middle = np.mean(positions,1)
    middle_direction = positions - middle[:,np.newaxis]
    velocities -= middle_direction * middle_strength
    positions += velocities


#Function: Animate the simulation
def animate(frame):
    update_sim(positions, velocities, middle_strength)
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
scatter = axes.scatter(positions[0,:], positions[1,:], marker="o", edgecolor="k", lw=0.5)
scatter

anime = animation.FuncAnimation(figure, animate, frames=50, interval=50)
plt.show()
