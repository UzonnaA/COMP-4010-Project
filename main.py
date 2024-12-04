# from train import train
# from run import run

# player, enemy = train(20)
# run(player, enemy)


from train import train
from run import run


# Train the shared DQN
sharedDQN = train(1, periodic_saving=True, load=True, device_name="cpu")

# Run the environment using the trained shared DQN

run(sharedDQN)

# Train the shared DQN
sharedDQN = train(1, periodic_saving=True, load=True, device_name="cpu")

# Run the environment using the trained shared DQN

run(sharedDQN)
