# from train import train
# from run import run

# player, enemy = train(20)
# run(player, enemy)


from train import train
from run import run
import argparse
import os
from network import DQN
import torch
from constants import GRID_HEIGHT, GRID_WIDTH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training (DQN) Agent for Farm Game for 4010 Final Project")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--run", action="store_true", help="Run the model")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to train the model")
    parser.add_argument("--load", type=bool, default=True, help="Option to load the model (defaults to True)")
    parser.add_argument("--periodic-saving", type=bool, default=True, help="Save the weights periodically to a folder called weights (defaults to True)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (defaults to cpu)")
    parser.add_argument("--running_epsilon", type=float, default=0.1, help="Epsilon value for evaluating agent behaviour in run mode")
    
    args = parser.parse_args()
    network = None
    
    if args.train:
        if(not os.path.exists("weights")):
            os.makedirs("weights")
        network = train(args.episodes, args.periodic_saving, args.load)
    
    if args.run:
        device = torch.device(args.device)
        if(not network): # if the network isn't already in memory, load it from saved weights
            network = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=6, device=device, epsilon=args.running_epsilon)
            if(os.path.exists("weights/model.pth")):
                print("Running with pre-trained model")
                network.load_state_dict(torch.load("weights/model.pth", weights_only=True))
        
        run(network)

