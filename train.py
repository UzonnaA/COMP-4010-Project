from constants import *
from enviro import *
from tqdm import tqdm
import numpy as np
import torch
from network import DQN

# Define the step function here to avoid circular dependency
def step(env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type):
    state = np.array(grid)
    playerAction, enemyAction = sharedDQN.act(state)
    playerReward = -1
    enemyReward = -1

    # Take player action
    grid[player_pos[1]][player_pos[0]] = previous_cell_type
    if playerAction == 5:
        farmland_x, farmland_y = player_pos[0], player_pos[1] + 1
        if (farmland_y, farmland_x) in env.farmland:
            playerReward += env.farmland[(farmland_y, farmland_x)] * 10
            enemyReward -= env.farmland[(farmland_y, farmland_x)] * 10

    player_pos = env.playerAct(player_pos, playerAction, grid)

    # Take enemy action
    grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
    farmland_x, farmland_y = enemy_pos[0], enemy_pos[1] + 1
    if (farmland_y, farmland_x) in env.farmland:
        enemyReward += env.farmland[(farmland_y, farmland_x)] * 10
        playerReward -= env.farmland[(farmland_y, farmland_x)] * 10
        del env.farmland[(farmland_y, farmland_x)]

    enemy_pos = env.enemyAct(enemy_pos, enemyAction, grid)

    # If 25 actions have been taken
    if env.growth_tick >= 25:
        env.grow_farmland()
        env.growth_tick = 0

    # Actions have been taken
    env.growth_tick += 1

    previous_cell_type = grid[player_pos[1]][player_pos[0]]

    # Set the player's current position
    grid[player_pos[1]][player_pos[0]] = PLAYER

    # Set the enemy's current position
    grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

    nextState = np.array(grid)

    return playerAction, enemyAction, False, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type

# Training function
def train(episodes):
    # Device setup
    device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     print("USING GPU")
    #     device = torch.device('cuda')

    print("TRAINING START")

    env = farmEnvironment()

    # Use a single shared DQN for both the player and the enemy
    sharedDQN = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=6, device=device)
    
    previous_cell_type = OPEN_SPACE
    for episode in range(episodes):
        grid, player_pos, enemy_pos = env.setup()
        done = False
        totalPlayerReward = 0
        totalEnemyReward = 0

        for i in tqdm(range(500), desc=f"Episode {episode+1}", unit="step", leave=False):
            # Call step function with the shared DQN
            playerAction, enemyAction, done, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type = step(
                env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type
            )
            
            # Store the experience in memory
            state = np.array(grid)
            sharedDQN.remember(state, playerAction, playerReward, nextState, done)
            sharedDQN.remember(state, enemyAction, enemyReward, nextState, done)

            # Train the shared DQN with replay
            sharedDQN.replay()

            # Update the total rewards
            state = nextState
            totalPlayerReward += playerReward
            totalEnemyReward += enemyReward

        print(f"Episode {episode+1}/{episodes} - Player Reward: {totalPlayerReward} - Player Epsilon: {sharedDQN.epsilon:.4f} - Enemy Reward: {totalEnemyReward}")

    return sharedDQN


