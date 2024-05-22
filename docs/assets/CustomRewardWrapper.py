from minigrid.core.world_object import Door, Key, Goal
from gymnasium import Wrapper


class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.key_was_picked = False
        self.door_was_unlocked = False


    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Custom reward logic
        # Check if the agent picked up a key
        if action == self.unwrapped.actions.pickup and self.unwrapped.carrying and isinstance(self.unwrapped.carrying, Key) and not self.key_was_picked:
            reward = 0.5  # Assign a reward for picking up the key
            self.key_was_picked = True

        # Check if the agent opened a door
        if action == self.unwrapped.actions.toggle:
            front_cell = self.unwrapped.grid.get(*self.unwrapped.front_pos)
            if front_cell and isinstance(front_cell, Door):
                if front_cell.is_locked and self.unwrapped.carrying and isinstance(self.unwrapped.carrying, Key) and not self.door_was_unlocked:
                    reward = 0.5  # Assign a reward for unlocking a door
                    self.door_was_unlocked =  True
                if not front_cell.is_locked:
                    reward = -0.25

        # Check if the agent reached the goal
        if terminated and self.unwrapped.agent_pos == (self.unwrapped.width - 2, self.unwrapped.height - 2):
            reward = 2  # Assign a reward for reaching the goal
        else:
            reward = -0.1  # Small penalty for each step

        return obs, reward, terminated, truncated, info