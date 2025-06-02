from train import *
from training_helpers import *

from run import run
import argparse
import os
from PPO import PPO_discrete
from DQN import DQN
import torch
from constants import GRID_HEIGHT, GRID_WIDTH
from utils import str2bool
from enviro import farmEnvironment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training (DQN) Agent for Farm Game for 4010 Final Project")
    parser.add_argument('mode', choices=['train', 'run', 'experiments'], help='Mode to run: train or trial')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo'], default='ppo', help='Algorithm to use for training')
    parser.add_argument('--graphs_folder', type=str, default='default_graphs', help='Folder to save graphs')
    parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')
    # parser.add_argument('--episodes', type=int, default=100, help='Max episodes to train on')
    parser.add_argument('--random_player', type=str2bool, default=False, help='Run training with random policy for player')
    parser.add_argument('--random_enemy', type=str2bool, default=False, help='Run training with random policy for enemy')

    parser.add_argument('--seed', type=int, default=209, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=5e5, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
    parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
    
    args = parser.parse_args()
    args.dvc = torch.device(args.dvc) # from str to torch.device
    network = None
    
    if args.mode == 'train':
        print(f'training for {args.Max_train_steps} training steps, saving every {args.save_interval} steps')
        if(not os.path.exists("model")):
            os.makedirs("model")
            
        env = farmEnvironment()
        if args.algorithm == 'ppo':
            args.state_dim = GRID_HEIGHT * GRID_WIDTH
            args.action_dim = 6
            args.agent_name = 'player'
            player = PPO_discrete(**vars(args))
            
            args.action_dim = 4
            args.agent_name = 'enemy'
            enemy = PPO_discrete(**vars(args))
            if args.Loadmodel:
                if not args.random_player:
                    player.load(args.ModelIdex)
                if not args.random_enemy:
                    enemy.load(args.ModelIdex)
            trainPPO(player, enemy, env, args.save_interval, args.T_horizon, args.Max_train_steps, args.random_player, args.random_enemy, step_fn=stepWithoutPenalty, reward_fn=rewardCalculationCustom, experiment_folder_name=args.graphs_folder)
            
        elif args.algorithm == 'dqn':
            
            sharedNetwork = DQN((GRID_HEIGHT, GRID_WIDTH), actionSize=6, device=args.dvc, epsilonMin=0.01, epsilon=1, agent_name='shared')
            if args.Loadmodel:
                sharedNetwork.load(args.ModelIdex)
            trainDQN(sharedNetwork, env, args.save_interval, args.Max_train_steps, args.random_player, args.random_enemy, step_fn=stepWithoutPenalty, reward_fn=rewardCalculationCustom, experiment_folder_name=args.graphs_folder)
            

            
        
    if args.mode == 'run':
        device = torch.device(args.dvc)
        env = farmEnvironment()

        args.agent_name = 'player'
        args.state_dim = GRID_HEIGHT * GRID_WIDTH
        args.action_dim = 6

        player = PPO_discrete(**vars(args))
        
        args.agent_name = 'enemy'
        args.action_dim = 4
        enemy = PPO_discrete(**vars(args))

        if(args.Loadmodel):
            if not args.random_player:
                player.load(args.ModelIdex)
            if not args.random_enemy:
                enemy.load(args.ModelIdex)     
        run(player, enemy, env)
    
    if args.mode == 'experiments':
        
        if not os.path.exists("experiments"):
            os.makedirs("experiments")
        print(f'running experiments with {args.algorithm} algorithm.')
        print(f'training steps: {args.Max_train_steps}')
        print(f'device: {args.dvc}')
        env = farmEnvironment()
        args.state_dim = GRID_HEIGHT * GRID_WIDTH

        
        '''full parameter grid for 18 experiments'''
        # player_enemy_random_params = [(True, False), (False, True), (True, True)]
        # step_functions = [stepWithPenalty, stepWithoutPenalty]
        # reward_functions = [rewardCalculationLinear, rewardCalculationQuartic, rewardCalculationCustom]
        
        '''partial parameter grid for 12 experiments'''
        # player_enemy_random_params = [(True, False), (False, True), (True, True)]
        # step_functions = [stepWithPenalty, stepWithoutPenalty]
        # reward_functions = [rewardCalculationLinear, rewardCalculationCustom] 

        '''minimal parameter grid for 6 experiments'''
        # experiment_params = [(True, False, stepWithPenalty, rewardCalculationCustom), (False, True, stepWithPenalty, rewardCalculationCustom), (True, True, stepWithPenalty, rewardCalculationCustom), (True, False, stepWithPenalty, rewardCalculationLinear), (False, True, stepWithPenalty, rewardCalculationLinear), (True, True, stepWithPenalty, rewardCalculationLinear)]
        
        #perform a control before running the experiments: random player and enemy
        args.action_dim = 6
        args.agent_name = 'player'
        player = PPO_discrete(**vars(args))
        
        args.action_dim = 4
        args.agent_name = 'enemy'
        enemy = PPO_discrete(**vars(args))
        trainPPO(player, enemy, env, args.save_interval, args.T_horizon, args.Max_train_steps, True, True, step_fn=stepWithoutPenalty, reward_fn=rewardCalculationCustom,experiment_folder_name='control_graphs')
        
        
        experiment_params = [(False, True, stepWithPenalty, rewardCalculationCustom), (True, False, stepWithPenalty, rewardCalculationCustom), (False, False, stepWithPenalty, rewardCalculationCustom), (False, True, stepWithoutPenalty, rewardCalculationLinear), (True, False, stepWithoutPenalty, rewardCalculationLinear), (False, False, stepWithoutPenalty, rewardCalculationLinear)]
        experiment_graph_names = ['farmer_training_mintime_formulation', 'zombie_training_mintime_formulation', 'both_training_mintime_formulation', 'farmer_training_linear_simple', 'enemy_training_linear_simple', 'both_training_linear_simple']
        
        if args.algorithm == 'ppo':
            for i in range(len(experiment_params)):
                '''reset the agents for each experiment'''
                args.action_dim = 6
                args.agent_name = 'player'
                player = PPO_discrete(**vars(args))
                
                args.action_dim = 4
                args.agent_name = 'enemy'
                enemy = PPO_discrete(**vars(args))
                trainPPO(player, enemy, env, args.save_interval, args.T_horizon, args.Max_train_steps, experiment_params[i][0], experiment_params[i][1], step_fn=experiment_params[i][2], reward_fn=experiment_params[i][3],experiment_folder_name=f'ppo_experiments/{experiment_graph_names[i]}')
        if args.algorithm =='dqn':
            for i in range(len(experiment_params)):
                sharedNetwork = DQN((GRID_HEIGHT, GRID_WIDTH), actionSize=6, device=args.dvc, epsilonMin=0.01, epsilon=1, agent_name='shared')
                trainDQN(sharedNetwork, env, args.save_interval, args.Max_train_steps, experiment_params[i][0], experiment_params[i][1], step_fn=experiment_params[i][2], reward_fn=experiment_params[i][3], experiment_folder_name=f'dqn_experiments/{experiment_graph_names[i]}')
            
            
            
        

