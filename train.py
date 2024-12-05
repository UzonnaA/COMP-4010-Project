from constants import *
from enviro import *
from tqdm import tqdm
import numpy as np
import torch
from network import DQN
import os
from PPO import PPO_discrete
import matplotlib.pyplot as plt

# Define the step function here to avoid circular dependency
def step(env, grid, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type):
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
            playerReward += env.farmland[(farmland_y, farmland_x)] * 10
            info["crop_stage_harvested"] = env.farmland[(farmland_y, farmland_x)]
            enemyReward -= env.farmland[(farmland_y, farmland_x)] * 10

    player_pos, info["crop_successfully_planted"] = env.playerAct(player_pos, playerAction, grid)

    # Take enemy action
    grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
    farmland_x, farmland_y = enemy_pos[0], enemy_pos[1] + 1
    if (farmland_y, farmland_x) in env.farmland:
        enemyReward += env.farmland[(farmland_y, farmland_x)] * 10
        info["crop_stage_destroyed"] = env.farmland[(farmland_y, farmland_x)]
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
    return nextState, playerReward, enemyReward, grid, player_pos, enemy_pos, previous_cell_type, info

def plot_multiple_lists(lists_of_values, x_label, y_label, legend_labels, filename):
    for values, label in zip(lists_of_values, legend_labels):
        plt.plot(values, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def trainPPO(player, enemy, env, save_interval, load, T_horizon, Max_train_steps, random_player=False, random_enemy=False): # PPO using XinJingHao's implementation
    episode_length = 500
    
    
    traj_lenth, total_steps = 0, 0
    episode = 0
    
        
    harvest_stats_by_stage = []
    enemy_rewards_per_episode = []
    player_rewards_per_episode = []
    destroyed_stats_by_stage = []
    planted_stats = []
    ruined_stats = []
    harvested_stats = []
    total_player_losses = []
    total_enemy_losses = []
    
    loss_per_learning_call = []
    while total_steps < Max_train_steps:
        cumulative_enemy_reward = 0
        cumulative_player_reward = 0
        stats = {
            "stages_of_crops_harvested": [0, 0, 0, 0, 0],
            "total_harvested": 0,
            "stages_of_crops_destroyed": [0, 0, 0, 0, 0],
            "total_crops_ruined": 0,
            "total_crops_planted": 0,
        }
        episode += 1
        s, player_pos, enemy_pos = env.setup()
        s = np.array(s)
        done = False
        previous_cell_type = OPEN_SPACE
        for i in tqdm(range(episode_length), desc=f"Episode {episode+1}", unit="step", leave=False):
            '''Interact with Env'''
            if(random_player):
                playerAction = np.random.randint(0, 6)
            else:
                playerAction, logprob_playerAction = player.select_action(s.flatten(), deterministic=False) # use stochastic when training
                
            if(random_enemy):
                enemyAction = np.random.randint(0, 4)
            else:
                enemyAction, logprob_enemyAction = enemy.select_action(s.flatten(), deterministic=False) # use stochastic when training
            
            nextState, playerReward, enemyReward, s, player_pos, enemy_pos, previous_cell_type, info = step(
                env, s, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type
            )
            done = i >= episode_length - 1
            
            if info["crop_stage_harvested"]:
                stats["stages_of_crops_harvested"][info["crop_stage_harvested"] - 1] += 1
                stats["total_harvested"] += 1
            if info["crop_stage_destroyed"]:
                stats["stages_of_crops_destroyed"][info["crop_stage_destroyed"] - 1] += 1
                stats["total_crops_ruined"] += 1
                
            stats["total_crops_planted"] += 1 if info["crop_successfully_planted"] else 0
            cumulative_enemy_reward += enemyReward
            cumulative_player_reward += playerReward
            
            '''Store the current transition'''
            if not random_player:
                player.put_data(s.flatten(), playerAction, playerReward, nextState.flatten(), logprob_playerAction, done, False, idx = traj_lenth) # dead&win is always false, there is no termination in our game
            if not random_enemy:
                enemy.put_data(s.flatten(), enemyAction, enemyReward, nextState.flatten(), logprob_enemyAction, done, False, idx = traj_lenth)
            s = np.array(nextState)

            traj_lenth += 1
            total_steps += 1

            '''Update if its time'''
            if traj_lenth % T_horizon == 0:
                if not random_player:
                    total_player_losses.append(player.train())
                if not random_enemy:
                    total_enemy_losses.append(enemy.train())
                traj_lenth = 0

            '''Save model'''
            if total_steps % save_interval==0:
                if not random_player:
                    player.save(total_steps)
                if not random_enemy:
                    enemy.save(total_steps)
        enemy_rewards_per_episode.append(cumulative_enemy_reward)
        player_rewards_per_episode.append(cumulative_player_reward)
        harvest_stats_by_stage.append(stats["stages_of_crops_harvested"])
        destroyed_stats_by_stage.append(stats["stages_of_crops_destroyed"])
        planted_stats.append(stats["total_crops_planted"])
        ruined_stats.append(stats["total_crops_ruined"])
        harvested_stats.append(stats["total_harvested"])
        
    harvest_stats_by_stage = np.array(harvest_stats_by_stage).T.tolist()
    destroyed_stats_by_stage = np.array(destroyed_stats_by_stage).T.tolist()
    
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 20
    harvested_stats = moving_average(harvested_stats, window_size)
    ruined_stats = moving_average(ruined_stats, window_size)
    planted_stats = moving_average(planted_stats, window_size)
    enemy_rewards_per_episode = moving_average(enemy_rewards_per_episode, window_size)
    player_rewards_per_episode = moving_average(player_rewards_per_episode, window_size)
    
    for i in range(5):
        harvest_stats_by_stage[i] = moving_average(harvest_stats_by_stage[i], window_size)
        destroyed_stats_by_stage[i] = moving_average(destroyed_stats_by_stage[i], window_size)
    
    if not random_player:
        total_player_losses = np.array(total_player_losses).flatten().tolist()
        total_player_losses = moving_average(total_player_losses, window_size)
        plot_multiple_lists([total_player_losses], "Training Steps (minibatch)", "Player Loss", ["Player Loss"], "player_loss.png")
        
    if not random_enemy:
        total_enemy_losses = np.array(total_enemy_losses).flatten().tolist()
        total_enemy_losses = moving_average(total_enemy_losses, window_size)
        plot_multiple_lists([total_enemy_losses], "Training Steps (minibatch)", "Enemy Loss", ["Enemy Loss"], "enemy_loss.png")
    
    plot_multiple_lists([enemy_rewards_per_episode], "Episodes", "Enemy Reward (Moving Average)", ["Enemy Reward"], "enemy_rewards.png")
    plot_multiple_lists([player_rewards_per_episode], "Episodes", "Player Reward (Moving Average)", ["Player Reward"], "player_rewards.png")
    plot_multiple_lists(harvest_stats_by_stage, "Episodes", "Number of crops harvested (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "harvested_stats_by_crop.png")
    plot_multiple_lists(destroyed_stats_by_stage, "Episodes", "Number of crops destroyed (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "destroyed_stats_by_crop.png")
    plot_multiple_lists([planted_stats], "Episodes", "Number of crops planted (Moving Average)", ["Planted"], "planted_stats.png")
    plot_multiple_lists([ruined_stats], "Episodes", "Number of crops destroyed by enemy (Moving Average)", ["Destroyed"], "destroyed_stats.png")
    plot_multiple_lists([harvested_stats], "Episodes", "Number of crops harvested by player (Moving Average)", ["Harvested"], "harvested_stats.png")
    
def trainSharedDQN(sharedDQN, env, save_interval, load, T_horizon, Max_train_steps): # PPO using XinJingHao's implementation
    episode_length = 500
    
    
    traj_lenth, total_steps = 0, 0
    episode = 0
    
    import matplotlib.pyplot as plt

    def plot_multiple_lists(lists_of_values, x_label, y_label, legend_labels, filename):
        for values, label in zip(lists_of_values, legend_labels):
            plt.plot(values, label=label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.savefig(filename)
        plt.close()
        
    harvest_stats_by_stage = []
    destroyed_stats_by_stage = []
    planted_stats = []
    ruined_stats = []
    harvested_stats = []
        
    while total_steps < Max_train_steps:
        stats = {
            "stages_of_crops_harvested": [0, 0, 0, 0, 0],
            "total_harvested": 0,
            "stages_of_crops_destroyed": [0, 0, 0, 0, 0],
            "total_crops_ruined": 0,
            "total_crops_planted": 0,
        }
        episode += 1
        s, player_pos, enemy_pos = env.setup()
        s = np.array(s)
        done = False
        previous_cell_type = OPEN_SPACE
        for i in tqdm(range(episode_length), desc=f"Episode {episode+1}", unit="step", leave=False):
            '''Interact with Env'''
            playerAction, logprob_playerAction = player.select_action(s.flatten(), deterministic=False) # use stochastic when training
            #enemyAction, logprob_enemyAction = enemy.select_action(s, deterministic=False) # use stochastic when training
            enemyAction = np.random.randint(0, 4)
            nextState, playerReward, enemyReward, s, player_pos, enemy_pos, previous_cell_type, info = step(
                env, s, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type
            )
            done = i >= episode_length - 1
            
            if info["crop_stage_harvested"]:
                stats["stages_of_crops_harvested"][info["crop_stage_harvested"] - 1] += 1
                stats["total_harvested"] += 1
            if info["crop_stage_destroyed"]:
                stats["stages_of_crops_destroyed"][info["crop_stage_destroyed"] - 1] += 1
                stats["total_crops_ruined"] += 1
                
            stats["total_crops_planted"] += 1 if info["crop_successfully_planted"] else 0
            
            '''Store the current transition'''
            player.put_data(s.flatten(), playerAction, playerReward, nextState.flatten(), logprob_playerAction, done, False, idx = traj_lenth) # dead&win is always false, there is no termination in our game
            #enemy.put_data(s, enemyAction, enemyReward, nextState, logprob_enemyAction, done, False, idx = traj_lenth)
            s = np.array(nextState)

            traj_lenth += 1
            total_steps += 1

            '''Update if its time'''
            if traj_lenth % T_horizon == 0:
                player.train()
                traj_lenth = 0

            '''Save model'''
            if total_steps % save_interval==0:
                player.save(total_steps)
        harvest_stats_by_stage.append(stats["stages_of_crops_harvested"])
        destroyed_stats_by_stage.append(stats["stages_of_crops_destroyed"])
        planted_stats.append(stats["total_crops_planted"])
        ruined_stats.append(stats["total_crops_ruined"])
        harvested_stats.append(stats["total_harvested"])
        
    harvest_stats_by_stage = np.array(harvest_stats_by_stage).T.tolist()
    destroyed_stats_by_stage = np.array(destroyed_stats_by_stage).T.tolist()
    
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 20
    harvested_stats = moving_average(harvested_stats, window_size)
    ruined_stats = moving_average(ruined_stats, window_size)
    planted_stats = moving_average(planted_stats, window_size)
    
    for i in range(5):
        harvest_stats_by_stage[i] = moving_average(harvest_stats_by_stage[i], window_size)
        destroyed_stats_by_stage[i] = moving_average(destroyed_stats_by_stage[i], window_size)

    plot_multiple_lists(harvest_stats_by_stage, "Episodes", "Number of crops harvested (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "harvested_stats_by_crop.png")
    plot_multiple_lists(destroyed_stats_by_stage, "Episodes", "Number of crops destroyed (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "destroyed_stats_by_crop.png")
    plot_multiple_lists([planted_stats], "Episodes", "Number of crops planted (Moving Average)", ["Planted"], "planted_stats.png")
    plot_multiple_lists([ruined_stats], "Episodes", "Number of crops destroyed by enemy (Moving Average)", ["Destroyed"], "destroyed_stats.png")
    plot_multiple_lists([harvested_stats], "Episodes", "Number of crops harvested by player (Moving Average)", ["Harvested"], "harvested_stats.png")
    

# Training function
# def train(episodes, periodic_saving, load, device_name):

#     # Device setup
#     device = torch.device(device_name)
#     # if torch.cuda.is_available():
#     #     print("USING GPU")
#     #     device = torch.device('cuda')

#     print("TRAINING START")

#     env = farmEnvironment()

#     # Use a single shared DQN for both the player and the enemy
#     sharedDQN = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=6, device=device)
    
#     if load and os.path.exists("weights/model.pth"):
#         print("Loading model")
#         sharedDQN.load_state_dict(torch.load("weights/model.pth", weights_only=True))
    
#     previous_cell_type = OPEN_SPACE
#     for episode in range(episodes):
#         grid, player_pos, enemy_pos = env.setup()
#         done = False
#         totalPlayerReward = 0
#         totalEnemyReward = 0

#         for i in tqdm(range(500), desc=f"Episode {episode+1}", unit="step", leave=False):
#             # Call step function with the shared DQN
#             nextState, playerReward, enemyReward, grid, player_pos, enemy_pos, previous_cell_type = step(
#                 env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type
#             )
            
#             # Store the experience in memory
#             state = np.array(grid)
#             # Train the shared DQN with replay
#             sharedDQN.replay()

#             # Update the total rewards
#             state = nextState
#             totalPlayerReward += playerReward
#             totalEnemyReward += enemyReward
            
#         if periodic_saving and (episode+1) % 5 == 0: # Save the model every 5 episodes
#             torch.save(sharedDQN.state_dict(), "weights/model.pth")
            
#         print(f"Episode {episode+1}/{episodes} - Player Reward: {totalPlayerReward} - Player Epsilon: {sharedDQN.epsilon:.4f} - Enemy Reward: {totalEnemyReward}")
    
#     torch.save(sharedDQN.state_dict(), "weights/model.pth")

#     return sharedDQN


