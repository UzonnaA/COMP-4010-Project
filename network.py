import random
import numpy as np
import torch
import torch.nn as nn

# DQN class definition
class DQN:
    def __init__(self, stateSize, actionSize):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.model = self.buildModel()
        self.memory = []
        self.gamma = 0.2
        self.epsilon = 1.0
        self.epsilonMin = 0.01
        self.epsilonDecay = 1
        self.batchSize = 1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def buildModel(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.stateSize[0] * self.stateSize[1], 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.actionSize)
        )

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.actionSize))
        state = torch.FloatTensor(state).unsqueeze(0)
        actValues = self.model(state)
        return torch.argmax(actValues).item()

    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batchSize:
            return
        miniBatch = random.sample(self.memory, self.batchSize)
        for (state, action, reward, nextState, done) in miniBatch:
            target = reward
            if not done:
                nextState = torch.FloatTensor(nextState).unsqueeze(0)
                target += self.gamma * torch.max(self.model(nextState)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            targetF = self.model(state).detach().numpy()
            targetF[0][action] = target
            self.model.zero_grad()
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.MSELoss()(output, torch.FloatTensor(targetF))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay