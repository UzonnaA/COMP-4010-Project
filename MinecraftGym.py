import gym
from gym import spaces
import numpy as np
import random
import pygame


# Define constants for grid elements
OPEN_SPACE = 0
FENCE = 1
PLAYER = 2
FARMLAND = 3
ENEMY = 4

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)      # Open space
RED = (255, 0, 0)        # Fences
BLUE = (0, 0, 255)       # Player
BROWN = (139, 69, 19)    # Farmland
PURPLE = (160, 32, 240)  # Enemy

# collidable objects for each agent
PLAYER_COLLISION = [FENCE, ENEMY]
ENEMY_COLLISION = [FENCE, PLAYER]

# Define actions
PLAYER_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FARM', 'STAY']
ENEMY_ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']

class MinecraftGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_width=20, grid_height=15, max_timesteps=1000):
        super(MinecraftGridEnv, self).__init__()
        
        # Grid dimensions
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.timestep = 0
        self.max_timesteps = max_timesteps  # Maximum number of timesteps before the episode ends

        self.player_standing_on = OPEN_SPACE
        # Action space: 4 directions + 1 action for farming + 1 action for staying
        self.action_space = spaces.Discrete(len(PLAYER_ACTIONS))
        self.enemy_action_space = spaces.Discrete(len(ENEMY_ACTIONS))
        
        #pygame variables
        self.pygame_initialized = False
        self.fps = 15
        self.cell_size = 40  # Size of each grid cell in pixels
        self.window_width = self.cell_size * self.grid_width
        self.window_height = self.cell_size * self.grid_height
        self.screen = None
        self.clock = pygame.time.Clock()

        
        # Observation space: each cell in the grid can be one of 5 types
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.grid_height, self.grid_width), dtype=np.int8)
        
        # Initialize the grid and positions
        self.reset()
    
    def reset(self):
        # Create grid and place fences around edges
        self.grid = None
        self.player_pos = None
        self.enemy_pos = None
        self.grid = np.full((self.grid_height, self.grid_width), OPEN_SPACE, dtype=np.int8)
        self.grid[0, :] = FENCE
        self.grid[-1, :] = FENCE
        self.grid[:, 0] = FENCE
        self.grid[:, -1] = FENCE
        
        # Place player at the center of the grid
        self.player_pos = [self.grid_width // 2, self.grid_height // 2]
        self.grid[self.player_pos[1], self.player_pos[0]] = PLAYER
        
        # Place enemy at a random open location
        while True:
            x, y = random.randint(1, self.grid_width - 2), random.randint(1, self.grid_height - 2)
            if self.grid[y, x] == OPEN_SPACE:
                self.enemy_pos = [x, y]
                self.grid[y, x] = ENEMY
                break

        # Return initial observation
        return self.grid
    

    def step(self, player_action, enemy_action):
        print(self.player_pos)
        print(self.enemy_pos)
        self.timestep += 1
        done = False
        # Handle player movement or farming action
        player_reward = 0
        enemy_reward = 0
        
        # Move the enemy
        player_reward = self.move_player(player_action)
        enemy_reward = self.move_enemy(enemy_action)
        print(player_reward)
        print(enemy_reward)

        if self.timestep > self.max_timesteps:
            done = True
            
        return self.grid, player_reward, enemy_reward, done, {}
    
    def is_valid_position(self, x, y, collision_set):
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height and self.grid[y, x] not in collision_set
    
    
    #determine new player position and return reward
    def move_player(self, action):
        reward = 0
        if PLAYER_ACTIONS[action] == 'FARM':
            farmland_x, farmland_y = self.player_pos[0], self.player_pos[1] + 1
            if farmland_y < self.grid_height and self.grid[farmland_y, farmland_x] == OPEN_SPACE:
                self.grid[farmland_y, farmland_x] = FARMLAND
        else:
            new_x, new_y = self.player_pos
            if PLAYER_ACTIONS[action] == 'UP':
                new_y -= 1
            elif PLAYER_ACTIONS[action] == 'DOWN':
                new_y += 1
            elif PLAYER_ACTIONS[action] == 'LEFT':
                new_x -= 1
            elif PLAYER_ACTIONS[action] == 'RIGHT':
                new_x += 1
            else:
                pass # STAY action

            if self.is_valid_position(new_x, new_y, PLAYER_COLLISION):
                self.grid[self.player_pos[1], self.player_pos[0]] = self.player_standing_on #restore the previous cell
                self.player_pos = [new_x, new_y]
                self.player_standing_on = self.grid[new_y, new_x] #store what the player was standing on before they move
                if(self.grid[new_y, new_x] == FARMLAND):
                    print("harvested")
                    reward += 1 #reward for harvesting
                self.grid[new_y, new_x] = PLAYER #they moved to a new position

        return reward
                
    def move_enemy(self, action):
        reward = 0
        new_x, new_y = self.enemy_pos
        if ENEMY_ACTIONS[action] == 'UP':
            new_y -= 1
        elif ENEMY_ACTIONS[action] == 'DOWN':
            new_y += 1
        elif ENEMY_ACTIONS[action] == 'LEFT':
            new_x -= 1
        elif ENEMY_ACTIONS[action] == 'RIGHT':
            new_x += 1

        if self.is_valid_position(new_x, new_y, ENEMY_COLLISION):
            self.grid[self.enemy_pos[1], self.enemy_pos[0]] = OPEN_SPACE
            self.enemy_pos = [new_x, new_y]
            if(self.grid[new_y, new_x] == FARMLAND):
                print("destroyed")
                reward += 1
            self.grid[new_y, new_x] = ENEMY
        return reward    
    def render(self, mode='human'):
        if(mode == 'human'):
            if not self.pygame_initialized:
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                pygame.display.set_caption("Grid environment")
                self.pygame_initialized = True
            self.draw_grid(self.screen, self.grid)
        else:
            print(self.grid)
    
    def draw_grid(self, screen, grid):
        screen.fill(WHITE)
        for y in range(len(self.grid)):
            for x in range(len(grid[0])):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                # Determine cell color based on grid type, with priority to farmland
                if self.grid[y][x] == FARMLAND:
                    color = BROWN
                elif self.grid[y][x] == PLAYER:
                    color = BLUE
                elif self.grid[y][x] == ENEMY:
                    color = PURPLE
                elif self.grid[y][x] == OPEN_SPACE:
                    color = GREEN
                elif self.grid[y][x] == FENCE:
                    color = RED
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border
        pygame.display.flip()




# Example usage
if __name__ == "__main__":
    env = MinecraftGridEnv()
    obs = env.reset()
    env.render()
    
    cumulative_p_reward = 0
    cumulative_e_reward = 0
    done = False
    while not done:
        env.clock.tick(env.fps)
        action = env.action_space.sample()  # Random action
        enemyaction = env.enemy_action_space.sample()
        pygame.event.pump()
        obs, player_reward, enemy_reward, done, info = env.step(action, enemyaction)
        
        cumulative_p_reward += player_reward
        cumulative_e_reward += enemy_reward
        
        env.render('human')
        print(f'Timestep: {env.timestep} Player reward: {player_reward}, Enemy reward: {enemy_reward}')
        print(f'Cumulative Player reward: {cumulative_p_reward}, Cumulative Enemy reward: {cumulative_e_reward}')