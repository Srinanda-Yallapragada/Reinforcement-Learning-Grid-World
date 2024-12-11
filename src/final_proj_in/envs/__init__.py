import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, render_mode=None):
        self.size = 10
        self.cold_val = 1
        self.window_size = 512
        #
        self.observation_space = spaces.Dict({
            "student": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            "goal": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
        })

        # 4 actions to take in the gridworld
        self.action_space = spaces.Discrete(4)
        self.actions = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.goal_states = [[9, 9], [4, 9],[9,7]]
        self.warm_buildings = [[2, 3], [2, 4], [4, 6], [2, 8], [7, 7], [7, 3]]
        self.snow = [[8, 0], [6, 0],[4,0]]
        self.road = [[x, 0] for x in range(10)]

        # These will be reset every time in the reset function
        self.student_location = np.array([0, 0])  # Starting position
        self.goal_location = np.array(random.choice(self.goal_states))
        
    # Helper method for formatting data
    def observation_data(self):
        return {"student": self.student_location, "goal": self.goal_location}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # These will be reset every time in the reset function
        self.student_location = np.array([0, 0])  # Starting position
        # self.goal_location = np.array([self.size - 1, self.size - 1])  # Goal position

        self.goal_location= np.array(random.choice(self.goal_states))
        observation = self.observation_data()
        self.cold_val=1

        if self.render_mode == "human":
            self.display()

        return observation, {}

    def step(self, action):

        # get action as a numpy array
        direction = self.actions[action]

        # action probabilities and next states
        if self.student_location.tolist() in self.snow:
            possible_next_states = [
                np.clip(self.student_location + direction, 0, self.size - 1),
                self.student_location,
            ]
            probabilities = [0.1,0.9]
            ind= np.random.choice(range(len(possible_next_states)), p=probabilities)
            self.student_location=possible_next_states[ind]

        else:
            self.student_location = np.clip(
                self.student_location + direction, 0, self.size - 1
            )

        # If reached terminal state, set to true
        reached_terminal_state = np.array_equal(self.student_location, self.goal_location)

        # for every step you are out in the cold, you get colder, this value is reset if you reach a warm building
        self.cold_val += 1

        # observed reward based on current location
        reward = self.reward_fn(reached_terminal_state)  

        observation = self.observation_data()

        if self.render_mode == "human":
            self.display()

        return observation, reward, reached_terminal_state, False, {}


    def reward_fn(self, reached_terminal_state):
        cold = self.cold_val
        agent_location = self.student_location.tolist()

        # Goal state
        if reached_terminal_state:
            return 100 - cold

        # Buildings
        if agent_location in self.warm_buildings:
            self.cold_val = 0
            return -0.1 - cold

        # Unused goal state
        if not reached_terminal_state :
            if agent_location in self.goal_states:
                return 10 - cold

        # Road
        if agent_location[1] == 0:
            return -0.5 - cold

        # Everywhere else
        else:
            return -2 - cold

    # Helper Methods for displaying data
    def render(self):
        if self.render_mode == "rgb_array":
            return self.display()

    def display(self):

        if self.render_mode=="human":
            # Initialize window
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            # Start clock
            if self.clock is None:
                self.clock = pygame.time.Clock()
        # create window
        surface = pygame.Surface((self.window_size, self.window_size))
        surface.fill((255, 255, 255))
        box = (self.window_size / self.size)  

        
        # Draw warm buildings
        for i in self.warm_buildings:
            i = np.array(i)
            pygame.draw.rect(surface, (128, 17, 9), pygame.Rect(box * i, (box, box)))

        # Draw second targets
        for i in self.goal_states:
            i = np.array(i)
            pygame.draw.rect(surface, (203, 243, 210), pygame.Rect(box * i,(box, box)))

        # Draw road
        for i in self.road:
            i = np.array(i)
            pygame.draw.rect(surface, (139, 146, 156), pygame.Rect(box * i,(box, box)))
            
        for i in self.snow:
            i = np.array(i)
            pygame.draw.rect(surface, (210, 200, 120), pygame.Rect(box * i,(box, box)))
       
       # Draw the target
        pygame.draw.rect(surface, (8, 81, 67), pygame.Rect(box * self.goal_location,(box, box)))
        # Draw agent
        pygame.draw.circle(surface, (0, 0, 0), (self.student_location + 0.5) * box, box/3)

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(surface, 0, (0, box * x), (self.window_size, box * x), width=1)
            pygame.draw.line(surface, 0, (box * x, 0), (box * x, self.window_size), width=1)

        if self.render_mode == "human":
            self.window.blit(surface, surface.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else: 
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()