from constants import *
from enviro import *
from run import step
from tqdm import tqdm
import numpy as np
import torch
from network import DQN

# Device setup
device = torch.device('cpu')
# if torch.cuda.is_available():
#     print("USING GPU")
#     device = torch.device('cuda')

def train(episodes):
    print("TRAINING START")

    env = farmEnvironment()

    player = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=5, device=device)
    enemy = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=4, device=device)
    
    previous_cell_type = OPEN_SPACE
    for episode in range(episodes):
        grid, player_pos, enemy_pos = env.setup()
        done = False
        totalPlayerReward = 0
        totalEnemyReward = 0

        for i in tqdm(range(1000), desc=f"Episode {episode+1}", unit="step", leave=False):
            playerAction, enemyAction, done, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, player, enemy, previous_cell_type = step(env, grid, player_pos, enemy_pos, player, enemy, previous_cell_type)
            
            state = np.array(grid)
            player.remember(state, playerAction, playerReward, nextState, done)
            enemy.remember(state, enemyAction, enemyReward, nextState, done)

            player.replay()
            enemy.replay()

            state = nextState
            totalPlayerReward += playerReward
            totalEnemyReward += enemyReward

        print(f"Episode {episode+1}/{episodes} - Player Reward: {totalPlayerReward} - Player Epsilon: {player.epsilon:.4f} - Enemy Reward: {totalEnemyReward} - Enemy Epsilon: {enemy.epsilon:.4f}")

    return player, enemy