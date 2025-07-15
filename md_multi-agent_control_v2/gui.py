#######################################################################
# gui.py
#
# GUI code for interactive control
#
# To Do: Fix Repulsive and Goal Forces, Fix Crystallization Saturation,
# Fix Visualization for temperature change, 
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
max_speed = 10

#Lennard-Jones Interaction Coefficients
epsilon = 7
sigma = 7
cutoff = 3 * sigma
optimal_range = (0.9 * sigma, 1.1 * sigma)
#Gamma Control
c1_gamma = 50
c2_gamma = 0.5 * np.sqrt(c1_gamma)
NUM_GOALS = 3
GOAL_RADIUS = 20
gamma_pos = [np.random.randint(100, 500, size=2) for _ in range(NUM_GOALS)]
discovered_goals = [False] * NUM_GOALS
active_goals = [True] * NUM_GOALS


#Obstacles
NUM_OBSTACLES = 4
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
friction = 0.02
mass = 1.0
kB = 1.0
temp = 10.0


#Agent specific Cooling
cooling_radius = 75 / PIXELS_PER_UNIT
cooling_radius_px = int(cooling_radius * PIXELS_PER_UNIT)
min_temp = 0.05

GOAL_AGENT_LIMIT = 5  # Max agents allowed within cooling radius of the goal

counter = 1

############################
# Init Agents aaaa
############################
md_sys = multi_agent(agents)

############################
# Exploration Set-up
############################
DISCOVERY_RES = 5  # resolution of each discovery cell in pixels
GRID_WIDTH = SIM_BOUNDS[1] // DISCOVERY_RES
GRID_HEIGHT = SIM_BOUNDS[1] // DISCOVERY_RES

discovery_grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=bool)
discovery_radius = 20  # in pixels

############################
# Object Setup
############################

pygame.init()

# Setup window and manager
window_size = (900, 700)
window_surface = pygame.display.set_mode(window_size)
manager = pygame_gui.UIManager(window_size)

#Lennard Jones Coeffs
lj_title = pygame_gui.elements.UILabel(relative_rect=pygame.Rect(((625, 135),(250, 25))),
                                         text= "Lennard Jones Coefficients:", manager=manager)

#Epsilon Slider
epsilon_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((625, 180), (250, 25)),
    start_value= epsilon,  # initial value
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
    start_value= sigma,  # initial value
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

#Agent Label
agent_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((625, 335), (250, 25)),
    text='Agents Value: 20',
    manager=manager
)

############################
# Agent Drawing Function
############################

def draw_agents(surface, agent_data, gamma_pos, agent_temps, obstacles, cooling_radius_px):

    # Draw agents
    for agent, temp_val in zip(agent_data, agent_temps):
        x, y = agent[:2]
        screen_x = int(x * PIXELS_PER_UNIT)
        screen_y = int(y * PIXELS_PER_UNIT)
        temp_norm = (temp_val - min_temp) / (temp - min_temp) if (temp - min_temp) > 0 else 0
        temp_norm = max(0, min(1,temp_norm))

        red = int(255 * temp_norm)
        blue = int(255 * (1 - temp_norm))
        color = (red, 0, blue)
        
        pygame.draw.circle(surface, color, (screen_x, screen_y), 5)

    # Draw goal
    for goal in gamma_pos:
        gx, gy = goal
        pygame.draw.circle(surface, (255, 0, 0), (gx, gy), GOAL_RADIUS - 15)
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

    goal_sim_units = [np.array(g) / PIXELS_PER_UNIT for g in gamma_pos]
    agents_near_goal = [0] * NUM_GOALS

    for i, goal_sim in enumerate(goal_sim_units):
        if not discovered_goals[i]:
            for agent in md_sys.agents:
                dist = np.linalg.norm(agent[:2] - goal_sim)
                if dist < GOAL_RADIUS / PIXELS_PER_UNIT:
                    discovered_goals[i] = True
                    print(f"Goal {i} discovered and broadcast!")
                    break

    
    for agent in md_sys.agents:
        for g_idx, goal_sim in enumerate(goal_sim_units):
            if discovered_goals[g_idx] and active_goals[g_idx]:
                dist = np.linalg.norm(agent[:2] - goal_sim)
                if dist < cooling_radius:
                    agents_near_goal[g_idx] += 1

    for g_idx in range(NUM_GOALS):
        if discovered_goals[g_idx] and agents_near_goal[g_idx] >= GOAL_AGENT_LIMIT:
            active_goals[g_idx] = False

    #Langevin Thermostat
    agent_c1 = []
    agent_c2 = []
    agent_temps = []
    agent_apply_gamma = []
    agent_target_goal = []

    for agent in md_sys.agents:
        pos = agent[:2]
        local_temp = temp
        apply_gamma = False
        target_goal = None
        min_dist = float("inf")

        for g_idx, goal_sim in enumerate(goal_sim_units):
            if discovered_goals[g_idx] and active_goals[g_idx]:
                dist = np.linalg.norm(pos - goal_sim)
                if dist < min_dist:
                    min_dist = dist
                    target_goal = goal_sim

        if target_goal is not None:
            dist_to_goal = np.linalg.norm(pos - target_goal)
            if dist_to_goal < cooling_radius:
                cooling_factor = dist_to_goal / cooling_radius
                local_temp = max(min_temp, temp * cooling_factor)
                
            if dist_to_goal < 2 * cooling_radius or active_goals[g_idx]:
                apply_gamma = True
        
        c1 = np.exp(-friction * time_delta)
        c2 = np.sqrt((1 - c1**2) * kB * local_temp / mass)

        agent_c1.append(c1)
        agent_c2.append(c2)
        agent_temps.append(local_temp)
        agent_target_goal.append(target_goal)
        agent_apply_gamma.append(apply_gamma)

    #Update Agents
    forces = md_sys.compute_forces(
        cutoff=cutoff,
        epsilon=epsilon,
        sigma=sigma,
        gamma_pos=gamma_pos,
        c1_gamma=c1_gamma,
        c2_gamma=c2_gamma,
        obstacles=obstacles,
        repulsion_strength=100.0,
        apply_gamma=agent_apply_gamma,
        target_goals=agent_target_goal
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
