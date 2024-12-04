import pygame
import sys
import numpy as np
from constants import *
from train import train
from enviro import farmEnvironment

# Pygame setup
def run(sharedDQN):
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

        # Use the updated step function that is now in train.py
        from train import step  # Import step here to avoid circular import at the top
        playerAction, enemyAction, done, nextState, playerReward, enemyReward, env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type = step(
            env, grid, player_pos, enemy_pos, sharedDQN, previous_cell_type
        )
        # Draw everything
        screen.fill(WHITE)
        
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if grid[y][x] == FARMLAND:
                    cur_stage = env.farmland.get((y, x))
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
