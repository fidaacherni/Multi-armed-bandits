from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots

def softmax(x, tau=1):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """
    
    e_x = np.exp(np.array(x) / tau)
    return e_x / e_x.sum(axis=0)


def gradient_bandit(
    heroes: Heroes, 
    alpha: float, 
    use_baseline: bool = True,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Gradient Bandit action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param alpha: The learning rate.
    :param use_baseline: Whether or not use avg return as baseline.
    :return: 
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: TThe average of rewards up to step t. For example: If 
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    h = np.array([0]*num_heroes, dtype=float)  # init h (the logits)
    rew_record = []                            # Rewards at each timestep
    avg_ret_record = []                        # Average reward up to each timestep
    tot_reg_record = []                        # Total regret up to each timestep
    opt_action_record = []                     # Percentage of optimal actions selected
    
    reward_bar = 0
    total_rewards = 0
    total_regret = 0

    ######### WRITE YOUR CODE HERE
    # Define the optimal reward and optimal hero index based on true success probabilities
    true_success_probabilities = [hero['true_success_probability'] for hero in heroes.heroes]
    optimal_hero_index = np.argmax(true_success_probabilities)  # Index of optimal hero
    optimal_reward = true_success_probabilities[optimal_hero_index]  # Optimal reward
    #########

    for t in range(heroes.total_quests):
        # Calculate softmax probabilities from the logits
        action_probabilities = softmax(h)

        # Choose an action based on the softmax probabilities
        action = np.random.choice(num_heroes, p=action_probabilities)

        # The chosen hero attempts the quest
        reward = heroes.attempt_quest(action)
        rew_record.append(reward)

        # Update total rewards and calculate running average reward
        total_rewards += reward
        avg_ret_record.append(total_rewards / (t + 1))

        # Calculate regret: the difference between optimal and actual reward
        regret = optimal_reward - reward
        total_regret += regret
        tot_reg_record.append(total_regret)

        # Check if the chosen hero was the optimal one
        opt_action_record.append(1 if action == optimal_hero_index else 0)

        # Calculate baseline reward if applicable
        if use_baseline:
            reward_bar = total_rewards / (t + 1)

        # Update logits h
        for i in range(num_heroes):
            if i == action:
                h[i] += alpha * (reward - reward_bar) * (1 - action_probabilities[i])
            else:
                h[i] -= alpha * (reward - reward_bar) * action_probabilities[i]
        #########

    return rew_record, avg_ret_record, tot_reg_record, np.cumsum(opt_action_record) / (
        np.arange(1, heroes.total_quests + 1))

if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various alpha values with baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=True)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })
    
    save_results_plots(results_list, plot_title="Gradient Bandits (with Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_with_baseline.pdf')

    # Test various alpha values without baseline
    alpha_values = [0.05, 0.1, 2]
    results_list = []
    for alpha in alpha_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=gradient_bandit,
                                                                    alpha=alpha, use_baseline=False)
        results_list.append({
            "exp_name": f"alpha={alpha}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Gradient Bandits (without Baseline) Experiment Results On Various Alpha Values",
                       results_folder='results', pdf_name='gradient_bandit_various_alpha_values_without_baseline.pdf')
