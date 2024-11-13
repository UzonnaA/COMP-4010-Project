import pygame
import sys
import numpy as np
from constants import *
from train import *

# Pygame setup
def run(player, enemy):

    env = farmEnvironment()

    pygame.init()
    print("STARTING...")

    # Set up the display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2D Grid Environment with Farmland and Enemy")
    clock = pygame.time.Clock()

    # Main loop
    running = True
    # Track the type of cell under the player to restore it when they move
    previous_cell_type = OPEN_SPACE

    grid, player_pos, enemy_pos = env.setup()

    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        playerAction, enemyAction, done, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, player, enemy, previous_cell_type = step(env, grid, player_pos, enemy_pos, player, enemy, previous_cell_type)
        
        # Draw everything
        screen.fill(WHITE)
        
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if grid[y][x] == FARMLAND:
                    cur_stage = env.farmland.get((y,x))
                    if cur_stage == 1:
                        color = BROWN
                    elif cur_stage == 2:
                        color = DARKGOLDENROD
                    elif cur_stage == 3:
                        color = COPPER
                    elif cur_stage == 4:
                        color = TIGERSEYE
                    elif cur_stage == 5:
                        color = YELLOW
                elif grid[y][x] == PLAYER:
                    color = BLUE
                elif grid[y][x] == ENEMY:
                    color = PURPLE
                elif grid[y][x] == OPEN_SPACE:
                    color = GREEN
                elif grid[y][x] == FENCE:
                    color = RED
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


def step(env, grid, player_pos, enemy_pos, player, enemy, previous_cell_type):
    state = np.array(grid)
    playerAction = player.act(state)
    enemyAction = enemy.act(state)
    playerReward = -1
    enemyReward = -1

    # Take player action
    grid[player_pos[1]][player_pos[0]] = previous_cell_type
    if playerAction == 5:
        farmland_x, farmland_y = player_pos[0], player_pos[1] + 1
        # if farmland_y < GRID_HEIGHT and grid[farmland_y][farmland_x] == FARMLAND and grid[farmland_y][farmland_x] != ENEMY:
        if (farmland_y, farmland_x) in env.farmland:
            playerReward += env.farmland[(farmland_y,farmland_x)]*10
            enemyReward -= env.farmland[(farmland_y,farmland_x)]*10

    player_pos = env.playerAct(player_pos, playerAction, grid)
    done = False

    # Take enemy action
    grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE
    farmland_x, farmland_y = enemy_pos[0], enemy_pos[1] + 1
    # if farmland_y < GRID_HEIGHT and grid[farmland_y][farmland_x] == FARMLAND and grid[farmland_y][farmland_x] != PLAYER:
    if (farmland_y, farmland_x) in env.farmland:
        enemyReward += env.farmland[(farmland_y,farmland_x)]*10
        playerReward -= env.farmland[(farmland_y,farmland_x)]*10
        del env.farmland[(farmland_y,farmland_x)]

    enemy_pos = env.enemyAct(enemy_pos, enemyAction, grid)
    done = False

    # If 25 actions have been taken, 
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

    return playerAction, enemyAction, done, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, player, enemy, previous_cell_type