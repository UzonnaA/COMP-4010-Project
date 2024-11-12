from constants import *
from enviro import *

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
        
        farm_training_dict = {}
        growth_train_tick = 0

        for i in range(100):
            playerAction = player.act(state)
            enemyAction = enemy.act(state)

            # If 25 actions have been taken, 
            if growth_train_tick >= 25:
                env.grow_farmland(farm_training_dict)
                growth_train_tick = 0

            # Take player action
            grid[player_pos[1]][player_pos[0]] = previous_cell_type
            player_pos = env.playerAct(player_pos, playerAction, grid, farm_training_dict)
            playerReward = 0
            done = False

            # Take enemy action
            grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
            enemy_pos = env.enemyAct(enemy_pos, enemyAction, grid)
            enemyReward = 0
            done = False

            # Actions have been taken
            growth_train_tick += 1

            if player_pos == enemy_pos:
                playerReward = -100
                enemyReward = 100
                done = True

            previous_cell_type = grid[player_pos[1]][player_pos[0]]

            # Set the player's current position
            grid[player_pos[1]][player_pos[0]] = PLAYER

            # Set the enemy's current position
            grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

            nextState = np.array(grid)
            player.remember(state, playerAction, playerReward, nextState, done)
            enemy.remember(state, enemyAction, enemyReward, nextState, done)

            player.replay()
            enemy.replay()

            state = nextState
            totalPlayerReward += playerReward
            totalEnemyReward += enemyReward

        print(f"Episode {episode+1}/{episodes} - Player Reward: {totalPlayerReward} - Player Epsilon: {player.epsilon:.4f} - Enemy Reward: {totalEnemyReward} - Enemy Epsilon: {enemy.epsilon:.4f}")

    return player, enemy