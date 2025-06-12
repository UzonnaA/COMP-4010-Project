# COMP-4010 Project: Reinforcement Learning in Multi-Agent Worlds

This project was part of COMP-4010 and focused on exploring how reinforcement learning agents behave in multi-agent environments. We worked with algorithms like DQN and PPO to test how agents adapt, learn from rewards, and evolve their strategies over time.

We used this as a chance to dive into real training experiments, visualize learning performance, and explore how reward shaping impacts agent behavior.

---

## What This Project Covers

Inside the repo, you’ll find:

- Learning agents built with DQN and PPO
- Simulations where agents either compete or learn in parallel
- Custom reward functions like quartic-based rewards
- Training graphs and metrics to track how agents perform across experiments

---

## Project Structure

Here’s what each folder contains:

- `control_graphs/` — Results from control/baseline tests
- `default_graphs/` — Training runs using standard reward setups
- `double_agent_training/` — Multi-agent training and evaluations
- `dqn_experiments/` — All DQN training graphs and outcomes
- `ppo_experiments/` — PPO-based training results
- `ppo_solofarmer_training_graphs/`, `ppo_solozombie_training_graphs/` — Isolated single-agent tests
- `quartic_rewards_*` — Experiments using nonlinear (quartic) reward shaping
- `randomized_policy_graphs/` — Tests with agents using randomized strategies
- `submission/` — Final course submission material

---

## Tools and Libraries

| Area | Tool |
|------|------|
| Language | Python |
| RL Library | Stable Baselines3 |
| Visualization | Matplotlib |
| Environment | Custom multi-agent simulation environment |

---

## Key Takeaways

Throughout the project, we tracked performance using graphs and reward metrics to understand how agents adapted in different scenarios. It was especially interesting to see how non-standard reward functions changed agent behavior and learning speed.

This project gave us a deeper understanding of how reinforcement learning actually works when applied to dynamic, agent-driven environments.

---



## 🤝 Thanks & Credits

This project was developed for the COMP-4010 course.  
Nicholas Faubert, Owen Lucas, Riley Olsen, Uzonna Alexander 
