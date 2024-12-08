from constants import *
from enviro import *
from tqdm import tqdm
import numpy as np
from training_helpers import *

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def trainPPO(player, enemy, env, save_interval, T_horizon, Max_train_steps, random_player=False, random_enemy=False, step_fn=stepWithoutPenalty, reward_fn=rewardCalculationCustom, experiment_folder_name='graphs'): # PPO using XinJingHao's implementation
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
    total_player_actor_losses = []
    total_enemy_actor_losses = []
    total_player_critic_losses = []
    total_enemy_critic_losses = []
    cumulative_enemy_wins = [0]
    cumulative_player_wins = [0]
    while total_steps < Max_train_steps:
        cumulative_enemy_reward = 0 #reset all stats between episodes
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
            
            nextState, playerReward, enemyReward, s, player_pos, enemy_pos, previous_cell_type, info = step_fn(
                env, s, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type, reward_fn=reward_fn
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
                    actor_losses, critic_losses = player.train()
                    total_player_actor_losses.append(actor_losses)
                    total_player_critic_losses.append(critic_losses)
                if not random_enemy:
                    actor_losses, critic_losses = enemy.train()
                    total_enemy_actor_losses.append(actor_losses)
                    total_enemy_critic_losses.append(critic_losses)
                traj_lenth = 0

            '''Save model'''
            if total_steps % save_interval==0:
                if not random_player:
                    print("we saved the player model")
                    player.save(total_steps)
                if not random_enemy:
                    print("we saved the enemy model")
                    enemy.save(total_steps)
        enemy_rewards_per_episode.append(cumulative_enemy_reward)
        player_rewards_per_episode.append(cumulative_player_reward)
        harvest_stats_by_stage.append(stats["stages_of_crops_harvested"])
        destroyed_stats_by_stage.append(stats["stages_of_crops_destroyed"])
        planted_stats.append(stats["total_crops_planted"])
        ruined_stats.append(stats["total_crops_ruined"])
        harvested_stats.append(stats["total_harvested"])
        
        if cumulative_enemy_reward > cumulative_player_reward: #calculate agent win ratios
            cumulative_enemy_wins.append(cumulative_enemy_wins[-1] + 1)
            cumulative_player_wins.append(cumulative_player_wins[-1])
        elif cumulative_enemy_reward < cumulative_player_reward:
            cumulative_enemy_wins.append(cumulative_enemy_wins[-1])
            cumulative_player_wins.append(cumulative_player_wins[-1] + 1)
        
    harvest_stats_by_stage = np.array(harvest_stats_by_stage).T.tolist()
    destroyed_stats_by_stage = np.array(destroyed_stats_by_stage).T.tolist()
    

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
        total_player_actor_losses = np.array(total_player_actor_losses).flatten().tolist()
        total_player_actor_losses = moving_average(total_player_actor_losses, window_size)
        plot_multiple_lists([total_player_actor_losses], "Training Steps (minibatch)", "Player Actor Loss", ["Player Actor Loss"], "player_actor_loss.png", experiment_folder_name)
        total_player_critic_losses = np.array(total_player_critic_losses).flatten().tolist()
        total_player_critic_losses = moving_average(total_player_critic_losses, window_size)
        plot_multiple_lists([total_player_critic_losses], "Training Steps (minibatch)", "Player Critic Loss", ["Player Critic Loss"], "player_critic_loss.png", experiment_folder_name)
        
    if not random_enemy:
        total_enemy_actor_losses = np.array(total_enemy_actor_losses).flatten().tolist()
        total_enemy_actor_losses = moving_average(total_enemy_actor_losses, window_size)
        plot_multiple_lists([total_enemy_actor_losses], "Training Steps (minibatch)", "Enemy Actor Loss", ["Enemy Actor Loss"], "enemy_actor_loss.png", experiment_folder_name)
        total_enemy_critic_losses = np.array(total_enemy_critic_losses).flatten().tolist()
        total_enemy_critic_losses = moving_average(total_enemy_critic_losses, window_size)
        plot_multiple_lists([total_enemy_critic_losses], "Training Steps (minibatch)", "Enemy Critic Loss", ["Enemy Critic Loss"], "enemy_critic_loss.png", experiment_folder_name)
            
    plot_multiple_lists([enemy_rewards_per_episode], "Episodes", "Enemy Reward (Moving Average)", ["Enemy Reward"], "enemy_rewards.png", experiment_folder_name)
    plot_multiple_lists([player_rewards_per_episode], "Episodes", "Player Reward (Moving Average)", ["Player Reward"], "player_rewards.png", experiment_folder_name)
    plot_multiple_lists(harvest_stats_by_stage, "Episodes", "Number of crops harvested (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "harvested_stats_by_crop.png", experiment_folder_name)
    plot_multiple_lists(destroyed_stats_by_stage, "Episodes", "Number of crops destroyed (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "destroyed_stats_by_crop.png", experiment_folder_name)
    plot_multiple_lists([planted_stats], "Episodes", "Number of crops planted (Moving Average)", ["Planted"], "planted_stats.png", experiment_folder_name)
    plot_multiple_lists([ruined_stats], "Episodes", "Number of crops destroyed by enemy (Moving Average)", ["Destroyed"], "destroyed_stats.png", experiment_folder_name)
    plot_multiple_lists([harvested_stats], "Episodes", "Number of crops harvested by player (Moving Average)", ["Harvested"], "harvested_stats.png", experiment_folder_name)
    plot_multiple_lists([cumulative_enemy_wins[1:], cumulative_player_wins[1:]], "Episodes", "Cumulative Wins", ["Enemy Wins", "Player Wins"], "cumulative_wins.png", experiment_folder_name)

def trainDQN(sharedDQN, env, save_interval, Max_train_steps, random_player=False, random_enemy=False, step_fn=stepWithoutPenalty, reward_fn=rewardCalculationCustom, experiment_folder_name='graphs'): # PPO using XinJingHao's implementation
    episode_length = 500
    
    total_steps = 0
    episode = 0
    
        
    harvest_stats_by_stage = []
    enemy_rewards_per_episode = []
    player_rewards_per_episode = []
    destroyed_stats_by_stage = []
    planted_stats = []
    ruined_stats = []
    harvested_stats = []
    total_network_losses = []
    cumulative_enemy_wins = [0]
    cumulative_player_wins = [0]
    epsilon = []

    previous_cell_type = OPEN_SPACE
    while total_steps < Max_train_steps:
        cumulative_enemy_reward = 0 #reset all stats between episodes
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

        for i in tqdm(range(500), desc=f"Episode {episode+1}", unit="step", leave=False):
            # create action
            if(random_player and random_enemy):
                playerAction = np.random.randint(0, 6)
                enemyAction = np.random.randint(0, 4)
            elif(random_player):
                _, enemyAction = sharedDQN.act(s)
                playerAction = np.random.randint(0, 6)
            elif(random_enemy):
                playerAction, _ = sharedDQN.act(s)
                enemyAction = np.random.randint(0, 4)
            else:
                playerAction, enemyAction = sharedDQN.act(s)
            
            # Call step function with the shared DQN
            nextState, playerReward, enemyReward, s, player_pos, enemy_pos, previous_cell_type, info = step_fn(
                env, s, player_pos, enemy_pos, playerAction, enemyAction, previous_cell_type, reward_fn=reward_fn
            )
            done = i >= episode_length - 1
            epsilon.append(sharedDQN.epsilon)
            
            # Store the experience in memory
            sharedDQN.remember(s, playerAction, playerReward, nextState, done)
            sharedDQN.remember(s, enemyAction, enemyReward, nextState, done)
            total_steps += 2
            
            # Train the shared DQN with replay in batches for stability and speed
            if total_steps % 1000 == 0:
                iterations = 1000 // sharedDQN.batchSize
                for _ in range(iterations):
                    total_network_losses.append(sharedDQN.replayBatch())
                sharedDQN.clearMemory()
            
            if total_steps % save_interval == 0:
                sharedDQN.save(total_steps)
                
            if info["crop_stage_harvested"]:
                stats["stages_of_crops_harvested"][info["crop_stage_harvested"] - 1] += 1
                stats["total_harvested"] += 1
            # else:
            #     stats["stages_of_crops_harvested"][info["crop_stage_harvested"] - 1] += 0
            #     stats["total_harvested"] += 0
                
            if info["crop_stage_destroyed"]:
                stats["stages_of_crops_destroyed"][info["crop_stage_destroyed"] - 1] += 1
                stats["total_crops_ruined"] += 1
            # else:
            #     stats["stages_of_crops_destroyed"][info["crop_stage_destroyed"] - 1] += 0
            #     stats["total_crops_ruined"] += 0
                
            stats["total_crops_planted"] += 1 if info["crop_successfully_planted"] else 0
            cumulative_enemy_reward += enemyReward
            cumulative_player_reward += playerReward
            
            s = np.array(nextState)



        enemy_rewards_per_episode.append(cumulative_enemy_reward)
        player_rewards_per_episode.append(cumulative_player_reward)
        harvest_stats_by_stage.append(stats["stages_of_crops_harvested"])
        destroyed_stats_by_stage.append(stats["stages_of_crops_destroyed"])
        planted_stats.append(stats["total_crops_planted"])
        ruined_stats.append(stats["total_crops_ruined"])
        harvested_stats.append(stats["total_harvested"])
        
        if cumulative_enemy_reward > cumulative_player_reward: #calculate agent win ratios
            cumulative_enemy_wins.append(cumulative_enemy_wins[-1] + 1)
            cumulative_player_wins.append(cumulative_player_wins[-1])
        elif cumulative_enemy_reward < cumulative_player_reward:
            cumulative_enemy_wins.append(cumulative_enemy_wins[-1])
            cumulative_player_wins.append(cumulative_player_wins[-1] + 1)
        
    harvest_stats_by_stage = np.array(harvest_stats_by_stage).T.tolist()
    destroyed_stats_by_stage = np.array(destroyed_stats_by_stage).T.tolist()
    print(player_rewards_per_episode)
    

    window_size = 20
    harvested_stats = moving_average(harvested_stats, window_size)
    ruined_stats = moving_average(ruined_stats, window_size)
    planted_stats = moving_average(planted_stats, window_size)
    enemy_rewards_per_episode = moving_average(enemy_rewards_per_episode, window_size)
    player_rewards_per_episode = moving_average(player_rewards_per_episode, window_size)
    
    for i in range(5):
        harvest_stats_by_stage[i] = moving_average(harvest_stats_by_stage[i], window_size)
        destroyed_stats_by_stage[i] = moving_average(destroyed_stats_by_stage[i], window_size)
    
    if not (random_player and random_enemy):
        total_network_losses = np.array(total_network_losses).flatten().tolist()
        total_network_losses = moving_average(total_network_losses, window_size)
        plot_multiple_lists([total_network_losses], "Training Steps (minibatch)", "Shared DQN Loss", ["DQN Loss"], "shared_dqn_loss.png", experiment_folder_name)
                    
    plot_multiple_lists([enemy_rewards_per_episode], "Episodes", "Enemy Reward (Moving Average)", ["Enemy Reward"], "enemy_rewards.png", experiment_folder_name)
    plot_multiple_lists([player_rewards_per_episode], "Episodes", "Player Reward (Moving Average)", ["Player Reward"], "player_rewards.png", experiment_folder_name)
    plot_multiple_lists(harvest_stats_by_stage, "Episodes", "Number of crops harvested (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "harvested_stats_by_crop.png", experiment_folder_name)
    plot_multiple_lists(destroyed_stats_by_stage, "Episodes", "Number of crops destroyed (moving average grouped by type)", ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"], "destroyed_stats_by_crop.png", experiment_folder_name)
    plot_multiple_lists([planted_stats], "Episodes", "Number of crops planted (Moving Average)", ["Planted"], "planted_stats.png", experiment_folder_name)
    plot_multiple_lists([ruined_stats], "Episodes", "Number of crops destroyed by enemy (Moving Average)", ["Destroyed"], "destroyed_stats.png", experiment_folder_name)
    plot_multiple_lists([harvested_stats], "Episodes", "Number of crops harvested by player (Moving Average)", ["Harvested"], "harvested_stats.png", experiment_folder_name)
    plot_multiple_lists([cumulative_enemy_wins[1:], cumulative_player_wins[1:]], "Episodes", "Cumulative Wins", ["Enemy Wins", "Player Wins"], "cumulative_wins.png", experiment_folder_name)
    

