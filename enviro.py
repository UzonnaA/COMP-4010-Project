import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from network import DQN
from constants import *

class farmEnvironment:
    def __init__(self):
        self.farmland = {}
        self.growth_tick = 0

    def grow_farmland(self):
        # Progress the stages of growth 
        for crop in self.farmland.keys():
            if self.farmland[crop] < 5:
                self.farmland[crop] += 1
        return self.farmland

    def setup(self):
        self.farmland = {}
        grid = [[OPEN_SPACE for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        for x in range(GRID_WIDTH):
            grid[0][x] = FENCE
            grid[GRID_HEIGHT - 1][x] = FENCE
        for y in range(GRID_HEIGHT):
            grid[y][0] = FENCE
            grid[y][GRID_WIDTH - 1] = FENCE

        player_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        grid[player_pos[1]][player_pos[0]] = PLAYER

        enemy_pos = [random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2)]
        while grid[enemy_pos[1]][enemy_pos[0]] != OPEN_SPACE:
            enemy_pos = [random.randint(1, GRID_WIDTH - 2), random.randint(1, GRID_HEIGHT - 2)]
        grid[enemy_pos[1]][enemy_pos[0]] = ENEMY

        return grid, player_pos, enemy_pos

    def playerAct(self, pos, action, grid):
        newX, newY = pos[0], pos[1]
        if action == 0 and pos[1] > 0: newY -= 1
        elif action == 1 and pos[1] < GRID_HEIGHT - 1: newY += 1
        elif action == 2 and pos[0] > 0: newX -= 1
        elif action == 3 and pos[0] < GRID_WIDTH - 1: newX += 1
        elif action == 4:
            farmland_x, farmland_y = pos[0], pos[1] + 1
            if farmland_y < GRID_HEIGHT and grid[farmland_y][farmland_x] == OPEN_SPACE:
                grid[farmland_y][farmland_x] = FARMLAND
                self.farmland[(farmland_y,farmland_x)] = 1
        if grid[newY][newX] != FENCE:
            return [newX, newY]
        return pos

    def enemyAct(self, pos, action, grid):
        newX, newY = pos[0], pos[1]
        if action == 0 and pos[1] > 0: newY -= 1
        elif action == 1 and pos[1] < GRID_HEIGHT - 1: newY += 1
        elif action == 2 and pos[0] > 0: newX -= 1
        elif action == 3 and pos[0] < GRID_WIDTH - 1: newX += 1
        if grid[newY][newX] != FENCE:
            return [newX, newY]
        return pos
