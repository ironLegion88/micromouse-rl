import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MicromouseEnv(gym.Env):
    def __init__(self):
        super(MicromouseEnv, self).__init__()
        # Hardcoded 4x4 maze (1 = wall, 0 = path)
        self.maze = np.array([
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 0]
        ])
        self.start_pos = (1, 1)  # Starting position
        self.goal_pos = (3, 2)   # Center/goal
        self.pos = self.start_pos
        self.directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # Right, Up, Left, Down
        self.facing = 0  # 0 = Right, 1 = Up, 2 = Left, 3 = Down
        self.max_steps = 100
        self.step_count = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: forward, 1: turn left, 2: turn right
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # (x, y, front, left, right)

    def reset(self):
        self.pos = self.start_pos
        self.facing = 0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        x, y = self.pos
        front_dir = self.directions[self.facing]
        left_dir = self.directions[(self.facing - 1) % 4]
        right_dir = self.directions[(self.facing + 1) % 4]
        
        front_wall = self._is_wall(x + front_dir[0], y + front_dir[1])
        left_wall = self._is_wall(x + left_dir[0], y + left_dir[1])
        right_wall = self._is_wall(x + right_dir[0], y + right_dir[1])
        
        return np.array([x / 3, y / 3, front_wall, left_wall, right_wall], dtype=np.float32)

    def _is_wall(self, x, y):
        if 0 <= x < 4 and 0 <= y < 4:
            return self.maze[x, y]
        return 1  # Out of bounds = wall

    def step(self, action):
        self.step_count += 1
        reward = -1  # Default step penalty
        
        if action == 0:  # Move forward
            next_x = self.pos[0] + self.directions[self.facing][0]
            next_y = self.pos[1] + self.directions[self.facing][1]
            if not self._is_wall(next_x, next_y):
                self.pos = (next_x, next_y)
        elif action == 1:  # Turn left
            self.facing = (self.facing - 1) % 4
        elif action == 2:  # Turn right
            self.facing = (self.facing + 1) % 4

        done = self.pos == self.goal_pos or self.step_count >= self.max_steps
        if self.pos == self.goal_pos:
            reward = 100

        return self._get_obs(), reward, done, {}

    def render(self):
        # Optional: Add Pygame visualization later
        print(f"Pos: {self.pos}, Facing: {self.facing}")
        print(self.maze)

# Test the environment
if __name__ == "__main__":
    env = MicromouseEnv()
    obs = env.reset()
    print("Initial observation:", obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break