#######################################################################
# This 3-D Flocking Simulation is based off of the original
# Craig Reynolds Paper and Algorithm. There are a ton like it.
# However, this one is 3-D which makes it alittle cooler to see. 
# Enjoy!
#
# Author: Humzah Durrani
#
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
boid_count = 200

#Margin and Turn Bound
margin = 200
turn_factor = 1

#Random Starting Positon range (X,Y,Z)
lower_pos = np.array([100,100,100])
upper_pos = np.array([1000,1500,1700])

#Velocity Range (X,Y,Z)
lower_vel = np.array([-5,-5,-3])
upper_vel = np.array([5,5,3])

#Cohesion: Force Applied to want to move to the center
middle_strength = 0.0005

#Separation: Alert Distance
alert_dist = 300

#Alignment: Match Speed 
flying_dist = 600
flying_str = 0.1

#Visualtion parameters:
trail_length = 20
pos_history = []

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


#Function: Animate the simulation
def animate(frame):
    global positions, velocities
    positions, velocities = update_sim(positions, velocities, middle_strength, alert_dist, flying_dist, flying_str, margin, turn_factor)
    pos_history.append(positions.copy())
    if len(pos_history) > trail_length:
        pos_history.pop(0)
    scatter._offsets3d = (positions[0], positions[1], positions [2])

    #Trail lines:
    for line in trail_lines:
        line.remove()
    trail_lines.clear()

    for i in range(boid_count):
        trail = np.array([p[:,i] for p in pos_history])
        line, = axes.plot(trail[:,0], trail[:,1], trail[:,2], alpha=0.3, linewidth=1)
        trail_lines.append(line)


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
trail_lines = []
scatter = axes.scatter(positions[0,:], positions[1,:], positions[2,:], marker="P", edgecolor="k", lw=0.5)
scatter
axes.set_xlim(0, sim_limits[0])
axes.set_ylim(0, sim_limits[1])
axes.set_zlim(0, sim_limits[2])
axes.set_title("3-D Flocking Simulation")

anime = animation.FuncAnimation(figure, animate, frames=50, interval=50, blit=False)
plt.show()

