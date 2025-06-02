import matplotlib.pyplot as plt
import numpy as np
from constants import *
from enviro import *
import os

def rewardCalculationLinear(stage): 
    return stage * 10

def rewardCalculationQuartic(stage): # requires normalization
    return (stage ** 4) * 10

def rewardCalculationCustom(stage): # balances minimum time task formulation
    match stage:
        case 1:
            return 0.0001
        case 2:
            return 0.001
        case 3:
            return 0.01
        case 4:
            return 0.1
        case 5:
            return 1
        case _:
            return None

# Step function with penalty for every action; minimum time task formulation
def stepWithPenalty(env, grid, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type, reward_fn=rewardCalculationLinear):
    state = np.array(grid)
    playerReward = -1
    enemyReward = -1
    info = {
        "crop_stage_harvested": None,
        "crop_successfully_planted": False,
        "crop_stage_destroyed": None,
    }
    # Take player action
    grid[player_pos[1]][player_pos[0]] = previous_cell_type
    if playerAction == 5:
        farmland_x, farmland_y = player_pos[0], player_pos[1] + 1
        if (farmland_y, farmland_x) in env.farmland:
            playerReward += reward_fn(env.farmland[(farmland_y, farmland_x)])
            info["crop_stage_harvested"] = env.farmland[(farmland_y, farmland_x)]
            enemyReward -= reward_fn(env.farmland[(farmland_y, farmland_x)])

    player_pos, info["crop_successfully_planted"] = env.playerAct(player_pos, playerAction, grid)

    # Take enemy action
    grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
    farmland_x, farmland_y = enemy_pos[0], enemy_pos[1] + 1
    if (farmland_y, farmland_x) in env.farmland:
        enemyReward += reward_fn(env.farmland[(farmland_y, farmland_x)])
        info["crop_stage_destroyed"] = env.farmland[(farmland_y, farmland_x)]
        playerReward -= reward_fn(env.farmland[(farmland_y, farmland_x)])
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
    return nextState, playerReward, enemyReward, grid, player_pos, enemy_pos, previous_cell_type, info

# Step function without penalty; maximizing task with unshaped reward
def stepWithoutPenalty(env, grid, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type, reward_fn=rewardCalculationLinear):
    state = np.array(grid)
    playerReward = 0
    enemyReward = 0
    info = {
        "crop_stage_harvested": None,
        "crop_successfully_planted": False,
        "crop_stage_destroyed": None,
    }
    # Take player action
    grid[player_pos[1]][player_pos[0]] = previous_cell_type
    if playerAction == 5:
        farmland_x, farmland_y = player_pos[0], player_pos[1] + 1
        if (farmland_y, farmland_x) in env.farmland:
            playerReward += reward_fn(env.farmland[(farmland_y, farmland_x)])
            info["crop_stage_harvested"] = env.farmland[(farmland_y, farmland_x)]

    player_pos, info["crop_successfully_planted"] = env.playerAct(player_pos, playerAction, grid)

    # Take enemy action
    grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
    farmland_x, farmland_y = enemy_pos[0], enemy_pos[1] + 1
    if (farmland_y, farmland_x) in env.farmland:
        enemyReward += reward_fn(env.farmland[(farmland_y, farmland_x)])
        info["crop_stage_destroyed"] = env.farmland[(farmland_y, farmland_x)]
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
    return nextState, playerReward, enemyReward, grid, player_pos, enemy_pos, previous_cell_type, info

def plot_multiple_lists(lists_of_values, x_label, y_label, legend_labels, filename, foldername):
    for values, label in zip(lists_of_values, legend_labels):
        plt.plot(values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if os.path.exists(foldername) == False:
        os.makedirs(foldername)
    plt.savefig(foldername + '/' + filename)
    plt.close()
    
def plot_two_y_axes(lists_of_values, x_label, y_labels, legend_labels, filename, foldername):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(lists_of_values[0], 'g-', label=legend_labels[0])
    ax2.plot(lists_of_values[1], 'b-', label=legend_labels[1])

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_labels[0], color='g')
    ax2.set_ylabel(y_labels[1], color='b')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    plt.savefig(os.path.join(foldername, filename))
    plt.close()
