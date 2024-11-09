import pygame
import sys
import random

import numpy as np
import torch
import torch.nn as nn

# Device setup
device = torch.device('cpu')
if torch.cuda.is_available():
    print("USING GPU")
    device = torch.device('cuda')

# DQN class definition
class DQN:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.model = self.buildModel()
        self.memory = []
        self.gamma = 0.2
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.999
        self.batchSize = 1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def buildModel(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.stateSize[0] * self.stateSize[1], 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.actionSize)
        )

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.actionSize))
        state = torch.FloatTensor(state).unsqueeze(0)
        actValues = self.model(state)
        return torch.argmax(actValues).item()

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batchSize:
            return
        miniBatch = random.sample(self.memory, self.batchSize)
        for (state, action, reward, nextState, done) in miniBatch:
            target = reward
            if not done:
                nextState = torch.FloatTensor(nextState).unsqueeze(0)
                target += self.gamma * torch.max(self.model(nextState)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            targetF = self.model(state).detach().numpy()
            targetF[0][action] = target
            self.model.zero_grad()
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.MSELoss()(output, torch.FloatTensor(targetF))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

# Pygame setup
pygame.init()
print("STARTING...")

# Constants
CELL_SIZE = 40
GRID_WIDTH = 20
GRID_HEIGHT = 15
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
PURPLE = (160, 32, 240)

# Cell types
OPEN_SPACE = 0
FENCE = 1
PLAYER = 2
FARMLAND = 3
ENEMY = 4

# Create grid
def create_grid(width, height):
    grid = [[OPEN_SPACE for _ in range(width)] for _ in range(height)]
    return grid

def grow_farmland(grid):
    new_grid = [row[:] for row in grid]
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == FARMLAND:
                if y > 0 and grid[y-1][x] == OPEN_SPACE:
                    new_grid[y-1][x] = FARMLAND
                if y < len(grid) - 1 and grid[y+1][x] == OPEN_SPACE:
                    new_grid[y+1][x] = FARMLAND
                if x > 0 and grid[y][x-1] == OPEN_SPACE:
                    new_grid[y][x-1] = FARMLAND
                if x < len(grid[0]) - 1 and grid[y][x+1] == OPEN_SPACE:
                    new_grid[y][x+1] = FARMLAND
    return new_grid

player = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=5)
enemy = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=4)

def setup():
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT)
    for x in range(GRID_WIDTH):
        grid[0][x] = FENCE
        grid[GRID_HEIGHT - 1][x] = FENCE
    for y in range(GRID_HEIGHT):
        grid[y][0] = FENCE
        grid[y][GRID_WIDTH - 1] = FENCE

    player_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
    grid[player_pos[1]][player_pos[0]] = PLAYER

    enemy_pos = [random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2)]
    while grid[enemy_pos[1]][enemy_pos[0]] != OPEN_SPACE:
        enemy_pos = [random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2)]
    grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

    return grid, player_pos, enemy_pos

def draw_grid(screen, grid):
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[y][x] == FARMLAND:
                color = BROWN
            elif grid[y][x] == PLAYER:
                color = BLUE
            elif grid[y][x] == ENEMY:
                color = PURPLE
            elif grid[y][x] == OPEN_SPACE:
                color = GREEN
            elif grid[y][x] == FENCE:
                color = RED
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

def playerAct(pos, action, grid):
    newX, newY = pos[0], pos[1]
    if action == 0 and pos[1] > 0: newY -= 1
    elif action == 1 and pos[1] < GRID_HEIGHT - 1: newY += 1
    elif action == 2 and pos[0] > 0: newX -= 1
    elif action == 3 and pos[0] < GRID_WIDTH - 1: newX += 1
    elif action == 4:
        farmland_x, farmland_y = pos[0], pos[1] + 1
        if farmland_y < GRID_HEIGHT and grid[farmland_y][farmland_x] == OPEN_SPACE:
            grid[farmland_y][farmland_x] = FARMLAND
    if grid[newY][newX] != FENCE:
        return [newX, newY]
    return pos

def enemyAct(pos, action, grid):
    newX, newY = pos[0], pos[1]
    if action == 0 and pos[1] > 0: newY -= 1
    elif action == 1 and pos[1] < GRID_HEIGHT - 1: newY += 1
    elif action == 2 and pos[0] > 0: newX -= 1
    elif action == 3 and pos[0] < GRID_WIDTH - 1: newX += 1
    if grid[newY][newX] != FENCE:
        return [newX, newY]
    return pos

def train(episodes):
    print("TRAINING START")
    previous_cell_type = OPEN_SPACE
    for episode in range(episodes):
        grid, player_pos, enemy_pos = setup()
        state = np.array(grid)
        done = False
        totalPlayerReward = 0
        totalEnemyReward = 0

        for i in range(100):
            playerAction = player.act(state)
            enemyAction = enemy.act(state)

            # Take player action
            grid[player_pos[1]][player_pos[0]] = previous_cell_type
            player_pos = playerAct(player_pos, playerAction, grid)
            playerReward = 0
            done = False

            # Take enemy action
            grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
            enemy_pos = enemyAct(enemy_pos, enemyAction, grid)
            enemyReward = 0
            done = False

            if player_pos == enemy_pos:
                playerReward = -100
                enemyReward = 100
                done = True

            previous_cell_type = grid[player_pos[1]][player_pos[0]]

            # Set the player's current position
            grid[player_pos[1]][player_pos[0]] = PLAYER

            # Set the enemy's current position
            grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

            nextState = np.array(grid)
            player.remember(state, playerAction, playerReward, nextState, done)
            enemy.remember(state, enemyAction, enemyReward, nextState, done)

            player.replay()
            enemy.replay()

            state = nextState
            totalPlayerReward += playerReward
            totalEnemyReward += enemyReward

        print(f"Episode {episode+1}/{episodes} - Player Reward: {totalPlayerReward} - Player Epsilon: {player.epsilon:.4f} - Enemy Reward: {totalEnemyReward} - Enemy Epsilon: {enemy.epsilon:.4f}")

train(500)

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Grid Environment with Farmland and Enemy")
clock = pygame.time.Clock()

# Main loop
running = True
# Track the type of cell under the player to restore it when they move
previous_cell_type = OPEN_SPACE

grid, player_pos, enemy_pos = setup()

while running:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = np.array(grid)
    playerAction = player.act(state)
    enemyAction = enemy.act(state)

    # Take player action
    grid[player_pos[1]][player_pos[0]] = previous_cell_type
    player_pos = playerAct(player_pos, playerAction, grid)
    playerReward = 0
    done = False

    # Take enemy action
    grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
    enemy_pos = enemyAct(enemy_pos, enemyAction, grid)
    enemyReward = 0
    done = False

    if player_pos == enemy_pos:
        playerReward = -100
        enemyReward = 100
        done = True

    previous_cell_type = grid[player_pos[1]][player_pos[0]]

    # Set the player's current position
    grid[player_pos[1]][player_pos[0]] = PLAYER

    # Set the enemy's current position
    grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

    nextState = np.array(grid)
    player.remember(state, playerAction, playerReward, nextState, done)
    enemy.remember(state, enemyAction, enemyReward, nextState, done)

    player.replay()
    enemy.replay()
    
    # Draw everything
    screen.fill(WHITE)
    draw_grid(screen, grid)
    pygame.display.flip()

pygame.quit()
sys.exit()
