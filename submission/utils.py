import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width, dtype=torch.float)
        self.l2 = nn.Linear(net_width, net_width, dtype=torch.float)
        self.l3 = nn.Linear(net_width, action_dim, dtype=torch.float)

    def forward(self, state):
        n = torch.tanh(self.l1(state + 1e-8))
        # else:
        #     print(state.cpu().detach().numpy())
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim=0):
        n = self.forward(state)  # Forward pass
        logits = self.l3(n)  # Compute logits
        # Apply stabilization trick before softmax
        # logits_stable = logits - torch.max(logits, dim=softmax_dim, keepdim=True).values
        prob = F.softmax(logits, dim=softmax_dim)
        if torch.any(torch.isclose(prob, torch.tensor(0.0, dtype=prob.dtype), atol=1e-8)):
            print("Probability contains zero value")
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width, dtype=torch.float)
        self.C2 = nn.Linear(net_width, net_width, dtype=torch.float)
        self.C3 = nn.Linear(net_width, 1, dtype=torch.float)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise