#######################################################################
# gui.py
#
# GUI code for interactive control
#
# To Do: Add Fog of War Exploration, Add better temperature dynamics,
# Add ability to add or remove agents, 
#
# Author: Humzah Durrani
#######################################################################


############################
# Import Libraries
############################

import pygame
import pygame_gui
import numpy as np
from md_multi_agent_policy import multi_agent


############################
# Starting Params
############################
SIM_BOUNDS = [0, 600]
PIXELS_PER_UNIT = 600 / (SIM_BOUNDS[1] - SIM_BOUNDS[0])

#Number of Agents
agents = 20

#Agent-Radius, Interaction Radius, Max-Speed
#agent_radius = 2.5
#interact_radius = 2.5
max_speed = 15

#Lennard-Jones Interaction Coefficients
epsilon = 10
sigma = 10
cutoff = 3 * sigma
optimal_range = (0.9 * sigma, 1.1 * sigma)
#Gamma Control
c1_gamma = 10
c2_gamma = 0.2 * np.sqrt(c1_gamma)
NUM_GOALS = 3
GOAL_RADIUS = 8  # Keep as is
gamma_pos = [np.random.randint(100, 500, size=2) for _ in range(NUM_GOALS)]

goal_found = [False] * NUM_GOALS


#Obstacles
NUM_OBSTACLES = 5
OBSTACLE_MIN_RADIUS = 20
OBSTACLE_MAX_RADIUS = 60

obstacles = [
    {
        "pos": np.random.randint(100, 500, size=2),
        "radius": np.random.randint(OBSTACLE_MIN_RADIUS, OBSTACLE_MAX_RADIUS)
    }
    for _ in range(NUM_OBSTACLES)
]

#Langevin Thermostat
friction = 0.1
mass = 1.0
kB = 1.0
temp = 1.0


#Agent specific Cooling
cooling_radius = 75 / PIXELS_PER_UNIT
cooling_radius_px = int(cooling_radius * PIXELS_PER_UNIT)
min_temp = 0.05

GOAL_AGENT_LIMIT = 10  # Max agents allowed within cooling radius of the goal

counter = 1


############################
# Init Agents
############################
md_sys = multi_agent(agents)


############################
# Exploration Set-up
############################
DISCOVERY_RES = 5  # resolution of each discovery cell in pixels
GRID_WIDTH = SIM_BOUNDS[1] // DISCOVERY_RES
GRID_HEIGHT = SIM_BOUNDS[1] // DISCOVERY_RES

discovery_grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=bool)
discovery_radius = 50  # in pixels

############################
# Object Setup
############################

pygame.init()

# Setup window and manager
window_size = (900, 700)
window_surface = pygame.display.set_mode(window_size)
manager = pygame_gui.UIManager(window_size)

#Goal Coordinate Text
#text_input = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(((0, 620),(250, 30))),
#                                         text= "Insert Goal Coordinates as 'x,y' ", manager=manager)

#Coordinate Input
#coord_input = pygame_gui.elements.UITextEntryLine(
#    relative_rect=pygame.Rect((10, 650), (200, 30)),
#    manager=manager
#)

#Coordinate Submit Button
#coord_button = pygame_gui.elements.UIButton(
#    relative_rect=pygame.Rect((225, 650), (100, 30)),
#    text='Set Goal',
#    manager=manager
#)

#Lennard Jones Coeffs
lj_title = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(((625, 135),(250, 25))),
                                         text= "Lennard Jones Coefficients:", manager=manager)

#Epsilon Slider
epsilon_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((625, 180), (250, 25)),
    start_value= 25,  # initial value
    value_range=(1.0, 50.0),  # min to max
    manager=manager
)

#Epsilon Label
epsilon_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((625, 160), (250, 25)),
    text='Epsilon Value: 10',
    manager=manager
)

#Sigma Slider
sigma_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((625, 220), (250, 25)),
    start_value= 25,  # initial value
    value_range=(1.0, 50.0),  # min to max
    manager=manager
)

#Sigma Label
sigma_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((625, 200), (250, 25)),
    text='Sigma Value: 2.5',
    manager=manager
)

#Gamma Coeff
gamma_title = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(((625, 255),(250, 25))),
                                         text= "Global Goal:", manager=manager)

#Gamma Slider
gamma_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((625, 300), (250, 25)),
    start_value= 25,  # initial value
    value_range=(1.0, 50.0),  # min to max
    manager=manager
)

#Gamma Label
gamma_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((625, 280), (250, 25)),
    text='Gamma Value: 25',
    manager=manager
)

#Gamma Slider
agent_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((625, 360), (250, 25)),
    start_value= 20,  # initial value
    value_range=(20, 100),  # min to max
    manager=manager
)

#Gamma Label
agent_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((625, 335), (250, 25)),
    text='Agents Value: 20',
    manager=manager
)



############################
# Agent Drawing Function
############################

def draw_agents(surface, agent_data, gamma_pos, agents_temps, obstacles, cooling_radius_px):
    #pygame.draw.rect(window_surface, (128, 0, 128), pygame.Rect(0, 0, 600, 600))

    # Draw agents
    for agent, temp_val in zip(agent_data, agent_temps):
        x, y = agent[:2]
        screen_x = int(x * PIXELS_PER_UNIT)
        screen_y = int(y * PIXELS_PER_UNIT)
        temp_norm = (temp_val - min_temp) / (temp - min_temp) if temp > min_temp else 0
        red = int(255 * temp_norm)
        blue = int(255 * (1 - temp_norm))
        color = (red, 0, blue)
        pygame.draw.circle(surface, color, (screen_x, screen_y), 5)

    # Draw goal
    for goal in gamma_pos:
        gx, gy = goal
        pygame.draw.circle(surface, (255, 0, 0), (gx, gy), GOAL_RADIUS)
        pygame.draw.circle(window_surface, (100, 255, 100), (gx, gy), cooling_radius_px, 1)

# Draw obstacles
    for obs in obstacles:
        ox, oy = obs["pos"]
        radius = obs["radius"]
        pygame.draw.circle(surface, (100, 100, 100), (ox, oy), radius)


############################
# GUI LOOP
############################

clock = pygame.time.Clock()
running = True

while running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle GUI events
        manager.process_events(event)

        #if event.type == pygame_gui.UI_BUTTON_PRESSED:
            #if event.ui_element == coord_button:
            #    raw_text = coord_input.get_text()
            #    try:
            #        raw_text = coord_input.get_text()
            #        coords = [int(val.strip()) for val in raw_text.split(',')]
            #        if len(coords) != 2:
            #            raise ValueError("Exactly two coordinates required.")
                    
                    # Result: integer array of shape (1, 2)
            #        goal_array = np.array(coords, dtype=int)
            #        gamma_pos = goal_array
            #        goal_found = False
            #        text_input.set_text(f"Goal: {goal_array}")
            #        print("Goal array shape:", goal_array.shape)
            #        print("Goal array value:", goal_array)

            #    except Exception as e:
            #        text_input.set_text(f"Invalid input: {e}")
            #        print("Error parsing input:", e)


        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:

            if event.ui_element == epsilon_slider:
                raw_val = epsilon_slider.get_current_value()
                epsilon_update = round(raw_val, 1) 
                if epsilon != epsilon_update:
                    epsilon = epsilon_update
                    epsilon_label.set_text(f"Epsilon Value: {epsilon_update}")

            if event.ui_element == sigma_slider:
                raw_val = sigma_slider.get_current_value()
                sigma_update = round(raw_val, 1)
                if sigma != sigma_update:
                    sigma = sigma_update
                    sigma_label.set_text(f"Sigma Value: {sigma_update}")

            if event.ui_element == gamma_slider:
                raw_val = gamma_slider.get_current_value()
                gamma_update = round(raw_val, 1)
                if c1_gamma != gamma_update:
                    c1_gamma = gamma_update
                    gamma_label.set_text(f"GammA Value: {gamma_update}")

            if event.ui_element == agent_slider:
                agent_update = agent_slider.get_current_value()
                if agents != agent_update:
                    agents = agent_update
                    agent_label.set_text(f"Agent Value: {agent_update}")

    manager.update(time_delta)
    #window_surface.fill((0, 0, 0))

    if goal_found and counter == 1:
        print("Goal Found!")
        counter += 1


    for i, goal in enumerate(gamma_pos):
        for agent in md_sys.agents:
            dist = np.linalg.norm(agent[:2] * PIXELS_PER_UNIT - goal)
            if dist < GOAL_RADIUS:
                goal_found[i] = True
                break


    #Langevin Thermostat
    goal_sim_units = [np.array(g) / PIXELS_PER_UNIT for g in gamma_pos]
    agent_c1 = []
    agent_c2 = []
    agent_temps = []
    agents_near_goal = sum( np.linalg.norm(agent[:2] - goal_sim_units) < cooling_radius
    for agent in md_sys.agents)

    for agent in md_sys.agents:
        dist_to_goal = np.linalg.norm(agent[:2] - goal_sim_units)
        local_temp = temp

        sees_goal = False
        if goal_found:
            if agents_near_goal < GOAL_AGENT_LIMIT:
                sees_goal = True
            elif dist_to_goal < cooling_radius:
                sees_goal = True  # part of crystallizing group

        # Apply cooling effect based on distance to goal
        if goal_found and dist_to_goal < cooling_radius:
            cooling_factor = dist_to_goal / cooling_radius  # 1 (far) to 0 (near)
            local_temp = max(min_temp, temp * cooling_factor)

        c1 = np.exp(-friction * time_delta)
        c2 = np.sqrt((1 - c1**2) * kB * local_temp / mass)

        agent_c1.append(c1)
        agent_c2.append(c2)
        agent_temps.append(local_temp)

    goal_force_on = goal_found and agents_near_goal < GOAL_AGENT_LIMIT
    
    #Update Agents
    forces = md_sys.compute_forces(
        cutoff=cutoff,
        epsilon=epsilon,
        sigma=sigma,
        gamma_pos=gamma_pos,
        c1_gamma=c1_gamma if goal_force_on else 0,
        c2_gamma=c2_gamma if goal_force_on else 0,
        obstacles=obstacles
    )

    md_sys.update(forces, max_speed=max_speed, c1_lang = agent_c1, c2_lang = agent_c2, mass = mass)

    # Clear and draw on main window surface
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            color = (255, 255, 0) if discovery_grid[x, y] else (128, 0, 128)  # Yellow or Purple
            rect = pygame.Rect(x * DISCOVERY_RES, y * DISCOVERY_RES, DISCOVERY_RES, DISCOVERY_RES)
            pygame.draw.rect(window_surface, color, rect)

    for agent in md_sys.agents:
        x, y = (agent[:2] * PIXELS_PER_UNIT).astype(int)
        gx = x // DISCOVERY_RES
        gy = y // DISCOVERY_RES
        rad_cells = discovery_radius // DISCOVERY_RES

        for dx in range(-rad_cells, rad_cells + 1):
            for dy in range(-rad_cells, rad_cells + 1):
                nx = gx + dx
                ny = gy + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    discovery_grid[nx, ny] = True

    # Draw agents and goal on the main surface
    draw_agents(window_surface, md_sys.agents, gamma_pos, agent_temps, obstacles, cooling_radius_px)
    pygame.draw.rect(window_surface, (30, 30, 30), pygame.Rect(600, 0, 300, 700))
    pygame.draw.rect(window_surface, (30, 30, 30), pygame.Rect(0 , 600, 700, 300))
    manager.draw_ui(window_surface)
    pygame.display.update()

pygame.quit()
