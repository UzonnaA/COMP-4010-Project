import random
import numpy as np
import torch
import torch.nn as nn

# DQN class definition
class DQN(nn.Module):
    def __init__(self, stateSize, actionSize, device, epsilonMin=0.01, epsilon=1):
        super(DQN, self).__init__()
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.device = device
        self.model = self.buildModel().to(device)
        self.memory = []
        self.gamma = 0.8
        self.epsilon = epsilon
        self.epsilonMin = epsilonMin
        self.epsilonDecay = 0.99991
        self.batchSize = 32
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    # def buildModel(self):
    #     return nn.Sequential(
    #         nn.Flatten(),
    #         nn.Linear(self.stateSize[0] * self.stateSize[1], 24),
    #         nn.ReLU(),
    #         nn.Linear(24, 24),
    #         nn.ReLU(),
    #         nn.Linear(24, self.actionSize)
    #     )

    def buildModel(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.stateSize[0] * self.stateSize[1], 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.actionSize * 2)  # Outputting actions for both agents
    )

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.choice(range(self.actionSize))
    #     state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     actValues = self.model(state)
    #     return torch.argmax(actValues).item()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action for both agents
            return (random.choice(range(self.actionSize)), random.choice(range(self.actionSize)))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actValues = self.model(state)
        # Split the output for farmer and enemy
        farmerActionValues, enemyActionValues = torch.split(actValues, self.actionSize, dim=1)
        return (torch.argmax(farmerActionValues).item(), torch.argmax(enemyActionValues).item())

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
            loss = nn.MSELoss()(output, targetF)

            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay