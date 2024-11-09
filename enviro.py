import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Define constants
CELL_SIZE = 40  # Size of each grid cell in pixels
GRID_WIDTH = 20  # Number of cells horizontally
GRID_HEIGHT = 15  # Number of cells vertically
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 30  

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)      # Open space
RED = (255, 0, 0)        # Fences
BLUE = (0, 0, 255)       # Player
BROWN = (139, 69, 19)    # Farmland Stage 1
DARKGOLDENROD = (160, 90, 23)    # Farmland Stage 2
COPPER = (191, 123, 42)    # Farmland Stage 3
TIGERSEYE = (222, 157, 0)    # Farmland Stage 4
YELLOW = (255, 215, 0)    # Farmland Stage 5
PURPLE = (160, 32, 240)  # Enemy

# Define cell types
OPEN_SPACE = 0
FENCE = 1
PLAYER = 2
FARMLAND = 3
ENEMY = 4

# Create the grid
def create_grid(width, height):
    grid = [[OPEN_SPACE for _ in range(width)] for _ in range(height)]
    return grid

# Initialize grid
grid = create_grid(GRID_WIDTH, GRID_HEIGHT)

# Place fences around the edge
for x in range(GRID_WIDTH):
    grid[0][x] = FENCE
    grid[GRID_HEIGHT - 1][x] = FENCE
for y in range(GRID_HEIGHT):
    grid[y][0] = FENCE
    grid[y][GRID_WIDTH - 1] = FENCE

# Place the player at the center
player_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
grid[player_pos[1]][player_pos[0]] = PLAYER

# Place the enemy randomly
enemy_pos = [random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2)]
while grid[enemy_pos[1]][enemy_pos[0]] != OPEN_SPACE:
    enemy_pos = [random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2)]
grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Grid Environment with Farmland and Enemy")
clock = pygame.time.Clock()

def draw_grid(screen, grid):
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # Determine cell color based on grid type, with priority to farmland
            if grid[y][x] == FARMLAND:
                cur_stage = farm_dict.get((y,x))
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
            pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border

def random_move(pos):
    direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
    new_x, new_y = pos[0], pos[1]
    
    if direction == 'LEFT':
        new_x -= 1
    elif direction == 'RIGHT':
        new_x += 1
    elif direction == 'UP':
        new_y -= 1
    elif direction == 'DOWN':
        new_y += 1
    
    if (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT and grid[new_y][new_x] != FENCE):
        return [new_x, new_y]
    return pos  # If move is invalid, stay in the same position

# Main loop
running = True
# Track the type of cell under the player to restore it when they move
previous_cell_type = OPEN_SPACE

# Farmland dictionary
farm_dict = {}
growth_tick = 0

while running:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player action with 10% chance to place farmland, 90% chance to move
    # This will be a reward system later
    action = random.choices(['move', 'farm'], weights=[90, 10])[0]

    if growth_tick >= 25:
        for crop in farm_dict.keys():
            if farm_dict[crop] < 5:
                farm_dict[crop] += 1
                growth_tick = 0

    
    if action == 'move':
        # Update the previous cell to the stored type (either OPEN_SPACE or FARMLAND)
        grid[player_pos[1]][player_pos[0]] = previous_cell_type
        player_pos = random_move(player_pos)
        # Store the cell type at the new player position
        previous_cell_type = grid[player_pos[1]][player_pos[0]]
        
    elif action == 'farm':
        # Attempt to place farmland one block south of the player's current position
        farmland_x, farmland_y = player_pos[0], player_pos[1] + 1
        if farmland_y < GRID_HEIGHT and grid[farmland_y][farmland_x] == OPEN_SPACE:
            grid[farmland_y][farmland_x] = FARMLAND
            farm_dict[(farmland_y,farmland_x)] = 1
            print("Placed farmland at:", (farmland_x, farmland_y))  # Debug statement to confirm farmland placement


    # Set the player's current position
    grid[player_pos[1]][player_pos[0]] = PLAYER

    # The player took the action
    growth_tick += 1

    # Move the enemy randomly
    new_enemy_pos = random_move(enemy_pos)

    ##! If the enemy destroys a crop, it cannot grow anymore
    if (enemy_pos[1],enemy_pos[0]) in farm_dict.keys():
        farm_dict.pop((enemy_pos[1],enemy_pos[0]), None)

    # The enemy can't clear the player
    if grid[enemy_pos[1]][enemy_pos[0]] != PLAYER:
        grid[enemy_pos[1]][enemy_pos[0]] = OPEN_SPACE  # Clear old enemy position
        enemy_pos = new_enemy_pos
        grid[enemy_pos[1]][enemy_pos[0]] = ENEMY  # Update new enemy position
    
    # Draw everything
    screen.fill(WHITE)
    draw_grid(screen, grid)
    pygame.display.flip()

pygame.quit()
sys.exit()
