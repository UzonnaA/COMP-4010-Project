import pygame
import sys

# Initialize Pygame
pygame.init()

# Define constants
CELL_SIZE = 40  # Size of each grid cell in pixels
GRID_WIDTH = 20  # Number of cells horizontally
GRID_HEIGHT = 15  # Number of cells vertically
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 30  # I guess we need this???

# Define colors
# These can be freely changed
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)      
RED = (255, 0, 0)        
BLUE = (0, 0, 255)       

# Define cell types
# We'll change these later
OPEN_SPACE = 0
FENCE = 1
PLAYER = 2

# Create the grid
def create_grid(width, height):
    grid = [[OPEN_SPACE for _ in range(width)] for _ in range(height)]
    return grid

# Initialize grid
grid = create_grid(GRID_WIDTH, GRID_HEIGHT)

# Place some fences around the edge
for x in range(GRID_WIDTH):
    grid[0][x] = FENCE
    grid[GRID_HEIGHT - 1][x] = FENCE
for y in range(GRID_HEIGHT):
    grid[y][0] = FENCE
    grid[y][GRID_WIDTH - 1] = FENCE

# Place the player at the center
player_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
grid[player_pos[1]][player_pos[0]] = PLAYER

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Grid Environment")
clock = pygame.time.Clock()

def draw_grid(screen, grid):
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[y][x] == OPEN_SPACE:
                color = GREEN
            elif grid[y][x] == FENCE:
                color = RED
            elif grid[y][x] == PLAYER:
                color = BLUE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border

# Main loop
running = True
while running:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # We won't need player movement for the real thing I don't think
        # But in case we do
        elif event.type == pygame.KEYDOWN:
            new_x, new_y = player_pos[0], player_pos[1]
    
        # Handle arrow keys and WASD keys
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:  # Left arrow or 'A'
                new_x -= 1
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:  # Right arrow or 'D'
                new_x += 1
            elif event.key == pygame.K_UP or event.key == pygame.K_w:  # Up arrow or 'W'
                new_y -= 1
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:  # Down arrow or 'S'
                new_y += 1
            
            # Check boundaries and fences
            if (0 <= new_x < GRID_WIDTH and
                0 <= new_y < GRID_HEIGHT and
                grid[new_y][new_x] != FENCE):
                
                # Update grid
                grid[player_pos[1]][player_pos[0]] = OPEN_SPACE
                player_pos = [new_x, new_y]
                grid[new_y][new_x] = PLAYER
    
    # Draw everything
    screen.fill(WHITE)
    draw_grid(screen, grid)
    pygame.display.flip()

pygame.quit()
sys.exit()
