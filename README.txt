run "pip install pygame" in the terminal

usage: main_options.py [-h] [--train] [--run] [--episodes EPISODES] [--load LOAD] [--periodic-saving PERIODIC_SAVING] [--device DEVICE] [--running_epsilon RUNNING_EPSILON]
EPISODES: integer [1<= EPISODES]
LOAD: boolean [True/False]
PERIODIC_SAVING: boolean [True/False]
DEVICE: string ["cpu"/"cuda"/"cuda0"]
RUNNING_EPSILON: float [0 < RUNNING_EPSILON <= 1]

main_options.py can be run with the following configurations:

python main_options.py --train ```trains with a default 20 episodes, running on cpu, automatically using the weights in model.pth and saving the model periodically every 5 episodes.```

python main_options.py --run ```renders a simulation of the episode, using the model.pth file if it has already been trained. runs with a baseline epsilon of .1.```

