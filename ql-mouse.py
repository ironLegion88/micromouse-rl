import numpy as np
import matplotlib.pyplot as plt
import random

# Random Maze Generator using Recursive Backtracking
def generate_maze(width=20, height=20):
    """Generate a solvable 20x20 maze using recursive backtracking."""
    # Initialize maze with walls (1) everywhere
    maze = np.ones((height, width), dtype=int)
    # Directions: Up, Right, Down, Left
    directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]

    def carve_path(x, y):
        """Recursively carve paths in the maze."""
        maze[x, y] = 0  # Mark current cell as path
        random.shuffle(directions)  # Randomize direction order
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < height and 0 <= new_y < width and maze[new_x, new_y] == 1):
                # Carve through the wall between current and new cell
                maze[x + dx // 2, y + dy // 2] = 0
                carve_path(new_x, new_y)

    # Start carving from (1, 1) to ensure outer walls remain
    carve_path(1, 1)
    
    # Set start and goal positions
    start_pos = (1, 1)  # Top-left corner (inner)
    goal_pos = (height - 2, width - 2)  # Bottom-right corner (inner)
    maze[start_pos] = 0
    maze[goal_pos] = 0
    
    return maze, start_pos, goal_pos

# Maze class to define the environment
class Maze:
    def __init__(self, maze_layout, start_pos, goal_pos):
        """Initialize maze with layout, start, and goal positions."""
        self.maze = np.array(maze_layout)
        self.height, self.width = self.maze.shape
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos)
        # Validate start and goal positions
        if not (0 <= self.start_pos[0] < self.height and 0 <= self.start_pos[1] < self.width):
            raise ValueError("Start position is outside maze boundaries.")
        if not (0 <= self.goal_pos[0] < self.height and 0 <= self.goal_pos[1] < self.width):
            raise ValueError("Goal position is outside maze boundaries.")
        if self.maze[self.start_pos] == 1 or self.maze[self.goal_pos] == 1:
            raise ValueError("Start or goal position is on a wall.")

    def show_maze(self, path=None):
        """Visualize the maze with optional path."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.maze, cmap='gray')
        plt.text(self.start_pos[1], self.start_pos[0], 'S', ha='center', va='center', color='red', fontsize=12)
        plt.text(self.goal_pos[1], self.goal_pos[0], 'G', ha='center', va='center', color='green', fontsize=12)
        if path:
            for row, col in path:
                plt.text(col, row, '#', ha='center', va='center', color='blue', fontsize=8)
        plt.xticks([]), plt.yticks([])
        plt.grid(color='black', linewidth=0.2)
        plt.show()

# Define possible actions: Up, Down, Left, Right
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (row_change, col_change)

class MicromouseAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, 
                 exploration_end=0.01, num_episodes=1000):
        """Initialize the Q-learning agent."""
        self.maze = maze
        self.q_table = np.zeros((maze.height, maze.width, len(ACTIONS)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, episode):
        """Calculate exploration rate with exponential decay."""
        return self.exploration_start * (self.exploration_end / self.exploration_start) ** (episode / self.num_episodes)

    def choose_action(self, state, episode):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() < self.get_exploration_rate(episode):
            return np.random.randint(len(ACTIONS))
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, next_state, reward):
        """Update Q-table using Q-learning formula."""
        best_next_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[state][action] = new_q

def run_episode(agent, maze, episode, train=True):
    """Simulate one episode of the Micromouse navigating the maze."""
    state = maze.start_pos
    path = [state]
    total_reward = 0
    steps = 0
    max_steps = maze.height * maze.width * 2  # Prevent infinite loops

    while steps < max_steps:
        action = agent.choose_action(state, episode)
        next_row = state[0] + ACTIONS[action][0]
        next_col = state[1] + ACTIONS[action][1]
        next_state = (next_row, next_col)

        # Check boundaries and walls
        if (next_row < 0 or next_row >= maze.height or 
            next_col < 0 or next_col >= maze.width or 
            maze.maze[next_row, next_col] == 1):
            reward = -10  # Wall penalty
            next_state = state
        elif next_state == maze.goal_pos:
            reward = 100  # Goal reward
            path.append(next_state)
            total_reward += reward
            steps += 1
            if train:
                agent.update_q_table(state, action, next_state, reward)
            break
        else:
            reward = -1  # Step penalty
            path.append(next_state)
            total_reward += reward
            steps += 1

        if train:
            agent.update_q_table(state, action, next_state, reward)
        state = next_state

    if steps >= max_steps:
        print(f"Episode terminated: Max steps ({max_steps}) reached.")
    return total_reward, steps, path

def train_agent(agent, maze, num_episodes):
    """Train the agent over multiple episodes."""
    rewards = []
    steps_list = []
    for episode in range(num_episodes):
        reward, steps, _ = run_episode(agent, maze, episode, train=True)
        rewards.append(reward)
        steps_list.append(steps)
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {reward}, Steps = {steps}")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_list)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.title('Training Steps Progress')
    plt.tight_layout()
    plt.show()
    
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)
    print(f"Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")
    return avg_reward, avg_steps

def test_agent(agent, maze, num_trials=10):
    """Test the trained agent and evaluate performance."""
    all_steps = []
    all_rewards = []
    optimal_path = None
    for trial in range(num_trials):
        reward, steps, path = run_episode(agent, maze, agent.num_episodes, train=False)
        all_steps.append(steps)
        all_rewards.append(reward)
        if trial == 0:  # Visualize the first trial
            optimal_path = path
            print(f"Trial {trial + 1} Path: {' -> '.join(map(str, path))}")
            maze.show_maze(path)
        print(f"Trial {trial + 1}: Steps = {steps}, Reward = {reward}")
    
    avg_steps = np.mean(all_steps)
    avg_reward = np.mean(all_rewards)
    print(f"\nTest Results over {num_trials} trials:")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    return avg_steps, avg_reward, optimal_path

# Function to find shortest path manually (for comparison)
def shortest_path(maze, start, goal):
    """Find shortest path using BFS for comparison."""
    from collections import deque
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        (row, col), path = queue.popleft()
        if (row, col) == goal:
            return len(path) - 1  # Exclude start position from step count
        for dr, dc in ACTIONS:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < maze.height and 0 <= new_col < maze.width and 
                maze.maze[new_row, new_col] == 0 and (new_row, new_col) not in visited):
                visited.add((new_row, new_col))
                queue.append(((new_row, new_col), path + [(new_row, new_col)]))
    return float('inf')  # No path found

# Main execution
if __name__ == "__main__":
    try:
        # Generate a random 20x20 maze
        maze_layout, start_pos, goal_pos = generate_maze(20, 20)
        maze = Maze(maze_layout, start_pos, goal_pos)
        agent = MicromouseAgent(maze, num_episodes=1000)  # Increased episodes for larger maze
        
        # Show initial maze
        print("Generated 20x20 Maze:")
        maze.show_maze()
        
        # Train the agent
        print("Training the Micromouse Agent...")
        train_agent(agent, maze, agent.num_episodes)
        
        # Test the agent
        print("\nTesting the Micromouse Agent...")
        test_agent(agent, maze, num_trials=10)
        
        # Compare with optimal path
        optimal_steps = shortest_path(maze, start_pos, goal_pos)
        print(f"\nShortest possible steps (BFS): {optimal_steps}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")