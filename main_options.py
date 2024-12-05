from train import trainPPO
from run import run
import argparse
import os
from PPO import PPO_discrete
import torch
from constants import GRID_HEIGHT, GRID_WIDTH
from utils import str2bool
from enviro import farmEnvironment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training (DQN) Agent for Farm Game for 4010 Final Project")
    parser.add_argument('mode', choices=['train', 'run'], help='Mode to run: train or trial')
    parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')
    # parser.add_argument('--episodes', type=int, default=100, help='Max episodes to train on')
    parser.add_argument('--random_player', type=str2bool, default=False, help='Run training with random policy for player')
    parser.add_argument('--random_enemy', type=str2bool, default=False, help='Run training with random policy for enemy')

    parser.add_argument('--seed', type=int, default=209, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=5e3, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e3, help='Model saving interval, in steps.')
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
        args.state_dim = GRID_HEIGHT * GRID_WIDTH
        args.action_dim = 6
        args.agent_name = 'player'
        player = PPO_discrete(**vars(args))
        
        args.action_dim = 4
        args.agent_name = 'enemy'
        enemy = PPO_discrete(**vars(args))
        network = trainPPO(player, enemy, env, args.save_interval, args.Loadmodel, args.T_horizon, args.Max_train_steps, args.random_player, args.random_enemy)
    
    if args.mode == 'run':
        device = torch.device(args.dvc)
        env = farmEnvironment()

        if(not network): # if the network isn't already in memory, load it from saved weights
            args.agent_name = 'player'
            args.state_dim = GRID_HEIGHT * GRID_WIDTH
            args.action_dim = 6

            player = PPO_discrete(**vars(args))
            
            args.agent_name = 'enemy'
            enemy = PPO_discrete(**vars(args))

            if(args.Loadmodel):
                player.load(args.ModelIdex)
                enemy.load(args.ModelIdex)        
        run(player, enemy, env)

