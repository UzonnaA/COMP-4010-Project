# 🧠 COMP-4010 Project: Reinforcement Learning in Multi-Agent Worlds

This project explores how reinforcement learning agents behave in complex multi-agent environments using algorithms like Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO). The goal is to understand how these agents learn, adapt, and perform in scenarios that involve both cooperation and competition.

---

## 🚀 What This Project Is About

Inside this repo, you’ll find:

- 🧠 **Learning Agents**: Agents trained using DQN and PPO to make decisions based on rewards and their environment.
- 🤖 **Multi-Agent Experiments**: Simulations where agents interact with each other—sometimes working together, sometimes against each other.
- 🎯 **Custom Rewards**: We tested different reward strategies, like quartic rewards, to see how they influence agent behavior.
- 📊 **Performance Tracking**: Training graphs and visualizations that show how well agents learn over time in various settings.

---

## 📁 What’s in the Repo

Each folder highlights a different experiment or scenario we tested:

- `control_graphs/`: Results from baseline control tests.
- `default_graphs/`: Standard training runs.
- `double_agent_training/`: Two-agent environments and training outcomes.
- `dqn_experiments/`: All the DQN training runs and visualizations.
- `ppo_experiments/`: PPO agent training experiments.
- `ppo_solofarmer_training_graphs/` & `ppo_solozombie_training_graphs/`: Single-agent PPO training results.
- `quartic_rewards_*`: Training outcomes when we modified the reward structure.
- `randomized_policy_graphs/`: Experiments using randomized agent policies.
- `submission/`: Final deliverables submitted for the course.

---

## 🛠️ Tech & Tools

| What | Tools |
|------|-------|
| Language | Python |
| RL Framework | Stable Baselines3 |
| Visualization | Matplotlib |
| Environment | Custom-built multi-agent simulation |

---

## 📈 What We Found

Each experiment folder includes graphs that show how our agents learned (or didn’t!). These visualizations helped us analyze training speed, reward optimization, and strategy development under different conditions.

---

## 🤝 Thanks & Credits

This project was developed for the COMP-4010 course.  
Nicholas Faubert, Owen Lucas, Riley Olsen, Uzonna Alexander 
