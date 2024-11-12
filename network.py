import random
import numpy as np
import torch
import torch.nn as nn

# DQN class definition
class DQN(nn.Module):
    def __init__(self, stateSize, actionSize, device):
        super(DQN, self).__init__()
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.device = device
        self.model = self.buildModel().to(device)
        self.memory = []
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.99
        self.batchSize = 64
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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
                nextState = torch.FloatTensor(nextState).unsqueeze(0).to(self.device)
                target += self.gamma * torch.max(self.model(nextState)).item()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            targetF = self.model(state).detach()
            targetF[0][action] = target
            self.model.zero_grad()
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.MSELoss()(output, torch.FloatTensor(targetF).to(self.device))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay