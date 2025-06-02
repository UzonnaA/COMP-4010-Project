# Constants
CELL_SIZE = 40
GRID_WIDTH = 20
GRID_HEIGHT = 15
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 30

# Colors
WHITE = (255, 255, 255)         # White
BLACK = (0, 0, 0)               # Black
GREEN = (0, 255, 0)             # Open Spaces
RED = (255, 0, 0)               # Fences
BLUE = (0, 0, 255)              # Player
PURPLE = (160, 32, 240)         # Enemy

# Farmland Colors
BROWN = (139, 69, 19)           # Farmland Stage 1
DARKGOLDENROD = (160, 90, 23)   # Farmland Stage 2
COPPER = (191, 123, 42)         # Farmland Stage 3
TIGERSEYE = (222, 157, 0)       # Farmland Stage 4
YELLOW = (255, 215, 0)          # Farmland Stage 5

# Cell types
OPEN_SPACE = 0
FENCE = 1
PLAYER = 2
FARMLAND = 3
ENEMY = 4