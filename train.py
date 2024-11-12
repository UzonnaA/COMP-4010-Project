from constants import *
from enviro import *
from run import step

# Device setup
device = torch.device('cpu')
if torch.cuda.is_available():
    print("USING GPU")
    device = torch.device('cuda')

def train(episodes):
    print("TRAINING START")

    env = farmEnvironment()

    player = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=5)
    enemy = DQN(stateSize=(GRID_HEIGHT, GRID_WIDTH), actionSize=4)
    
    previous_cell_type = OPEN_SPACE
    for episode in range(episodes):
        grid, player_pos, enemy_pos = env.setup()
        state = np.array(grid)
        done = False
        totalPlayerReward = 0
        totalEnemyReward = 0
        
        # farm_training_dict = {}
        growth_train_tick = 0

        for i in range(100):
            state = np.array(grid)

            playerAction, enemyAction, done, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, state, player, enemy, previous_cell_type = step(env, grid, player_pos, enemy_pos, state, player, enemy, previous_cell_type)
            
            player.remember(state, playerAction, playerReward, nextState, done)
            enemy.remember(state, enemyAction, enemyReward, nextState, done)

            player.replay()
            enemy.replay()

            state = nextState
            totalPlayerReward += playerReward
            totalEnemyReward += enemyReward

        print(f"Episode {episode+1}/{episodes} - Player Reward: {totalPlayerReward} - Player Epsilon: {player.epsilon:.4f} - Enemy Reward: {totalEnemyReward} - Enemy Epsilon: {enemy.epsilon:.4f}")

    return player, enemy