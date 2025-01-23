# Multi-armed-bandits
📜 Project Overview

This project explores multi-armed bandit (MAB) algorithms in a reinforcement learning setting, where Aragorn, the rightful heir to Gondor, must identify the greatest hero in Middle-earth to undertake dangerous quests. Each hero has an unknown probability of success in these trials, and the goal is to balance exploration and exploitation to find the best hero.

🚀 Implemented Algorithms

The project implements and compares several multi-armed bandit strategies:

- Epsilon-Greedy: Selects actions randomly with probability ϵ and chooses the best-known action otherwise.
- Upper Confidence Bound (UCB): Uses confidence bounds to balance exploration and exploitation dynamically.
- Boltzmann (Softmax) Exploration: Assigns action probabilities based on an exponential function of estimated values.
- Gradient Bandit Methods: Uses a preference-based learning approach to optimize action selection.

📊 Performance Analysis

Each algorithm is tested on hero trials, and performance is evaluated based on:

- Total Rewards: How well the method maximizes success.
- Regret: The difference between the obtained reward and the optimal strategy.
- Optimal Action Selection Rate: How often the best hero is chosen over time.

📂 Repository Structure

- heroes.py: Defines the Heroes class and manages success rates.
- eps_greedy.py: Implements the Epsilon-Greedy algorithm.
- ucb.py: Implements the Upper Confidence Bound (UCB) algorithm.
- boltzmann.py: Implements the Boltzmann (Softmax) exploration method.
- gradient_bandit.py: Implements the Gradient Bandits method.
- compare.py: Compares all the methods and tunes their parameters.

  
📈 Results & Findings


The Gradient Bandit method with optimal parameters (α = 1.55, baseline enabled) outperformed all other methods in terms of reward maximization and optimal action selection. The UCB method also performed well, while Boltzmann and Epsilon-Greedy showed mixed results depending on parameter tuning.
