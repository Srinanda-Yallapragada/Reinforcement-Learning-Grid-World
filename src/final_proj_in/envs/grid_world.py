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

        # for pygame rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # These will be reset every time in the reset function
        self.student_location = np.array([0, 0])  # Starting position
        self.goal_location = np.array([self.size - 1, self.size - 1])  # Goal position

        self.goal_states = [[9, 9], [9, 4],[4,9]]
        self.warm_buildings = [[2, 3], [2, 4], [4, 6], [2, 8], [7, 7], [7, 3]]
        self.snow = [[2, 5],[3,5],[3,4]]
        self.road = [[x, 0] for x in range(10)]

    def _get_obs(self):
        return {"student": self.student_location, "goal": self.goal_location}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # These will be reset every time in the reset function
        self.student_location = np.array([0, 0])  # Starting position
        # self.goal_location = np.array([self.size - 1, self.size - 1])  # Goal position

        self.goal_location= np.array(random.choice(self.goal_states))
        observation = self._get_obs()
        self.cold_val=1

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def step(self, action):

        # get action as a numpy array
        direction = self.actions[action]

        # student stays on the grid.
        # self.student_location = np.clip(
        #     self.student_location + direction, 0, self.size - 1
        # )
        if self.student_location.tolist() in self.snow:
            possible_next_states = [
                np.clip(self.student_location + direction, 0, self.size - 1),
                self.student_location,
            ]
            probabilities = [0.3, 0.7]
            ind= np.random.choice(range(len(possible_next_states)), p=probabilities)
            self.student_location=possible_next_states[ind]

        else:
            self.student_location = np.clip(
                self.student_location + direction, 0, self.size - 1
            )

        # If reached terminal state, set to true
        reached_terminal_state = np.array_equal(self.student_location, self.goal_location)

        # for every step you are out in the cold, you get colder
        # this value is reset if you reach a warm building
        self.cold_val += 1

        # reward = self.reward_fn()  # observed reward based on current location

        reward = self.reward_fn(reached_terminal_state)  # observed reward based on current location
        # reward = 1 if reached_terminal_state else 0  # Binary sparse rewards

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, reached_terminal_state, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels
        #
        # Draw warm buildings
        for i in self.warm_buildings:
            i = np.array(i)
            pygame.draw.rect(
                canvas,
                (255, 204, 153),
                pygame.Rect(
                    pix_square_size * i,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Draw secondary targets
        for i in self.goal_states:
            i = np.array(i)
            pygame.draw.rect(
                canvas,
                (0, 38, 153),
                pygame.Rect(
                    pix_square_size * i,
                    (pix_square_size, pix_square_size),
                ),
            )
        for i in self.road:
            i = np.array(i)
            pygame.draw.rect(
                canvas,
                (128, 128, 128),
                pygame.Rect(
                    pix_square_size * i,
                    (pix_square_size, pix_square_size),
                ),
            )
        for i in self.snow:
            i = np.array(i)
            pygame.draw.rect(
                canvas,
                (0, 250, 0),
                pygame.Rect(
                    pix_square_size * i,
                    (pix_square_size, pix_square_size),
                ),
            )
        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 153, 51),
            pygame.Rect(
                pix_square_size * self.goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.student_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # def reward_fn(self, reached_terminal_state):
    #     if reached_terminal_state:
    #         print("good")
    #         return 100
    #     if self.student_location.tolist() in self.goal_states:
    #         print("bad")
    #         return -100

    #     # by default -2 for all snow locations
    #     return -2

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
                print('unused')
                return -10 - cold

        # Road
        if agent_location[1] == 0:
            return -0.5 - cold

        # Everywhere else
        else:
            return -2 - cold
