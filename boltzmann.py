from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots


def boltzmann_policy(x, tau):
    """ Returns an index sampled from the softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: idx -- chosen index
    """

    ######### WRITE YOUR CODE HERE
    # Convert x to a numpy array
    x = np.array(x)

    # Compute softmax probabilities
    softmax_probabilities = np.exp(x / tau) / np.sum(np.exp(x / tau))

    # Sample an index based on the softmax probabilities
    index = np.random.choice(len(x), p=softmax_probabilities)
    return index


def boltzmann(
        heroes: Heroes,
        tau: float = 0.1,
        init_value: float = .0
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Boltzmann action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param tau: The temperature value (𝜏).
    :param init_value: Initial estimation of each hero's value.
    :return:
        - rew_record: The record of rewards at each timestep.
        - avg_ret_record: The average of rewards up to step t. For example: If
    we define `ret_T` = \sum^T_{t=0}{r_t}, `avg_ret_record` = ret_T / (1+T).
        - tot_reg_record: The total regret up to step t.
        - opt_action_record: Percentage of optimal actions selected.
    """

    num_heroes = len(heroes.heroes)
    values = [init_value] * num_heroes  # Initial action values
    rew_record = []  # Rewards at each timestep
    avg_ret_record = []  # Average reward up to each timestep
    tot_reg_record = []  # Total regret up to each timestep
    opt_action_record = []  # Percentage of optimal actions selected

    total_rewards = 0
    total_regret = 0

    ######### WRITE YOUR CODE HERE
    # Define the optimal reward and optimal hero index based on true success probabilities
    true_success_probabilities = [hero['true_success_probability'] for hero in heroes.heroes]
    optimal_hero_index = np.argmax(true_success_probabilities)  # Index of optimal hero
    optimal_reward = true_success_probabilities[optimal_hero_index]  # Optimal reward
    #########

    for t in range(heroes.total_quests):
        ######### WRITE YOUR CODE HERE
        # Use the Boltzmann policy to choose a hero
        action = boltzmann_policy(values, tau)

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

        # Update the value estimates for the chosen hero using incremental average
        values[action] += (reward - values[action]) / (t + 1)
        #########

    return rew_record, avg_ret_record, tot_reg_record, np.cumsum(opt_action_record) / (
        np.arange(1, heroes.total_quests + 1))


if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various tau values
    tau_values = [0.01, 0.1, 1, 10]
    results_list = []
    for tau in tau_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=boltzmann,
                                                                    tau=tau, init_value=0)

        results_list.append({
            "exp_name": f"tau={tau}",
            "reward_rec": rew_rec,
            "average_rew_rec": avg_ret_rec,
            "tot_reg_rec": tot_reg_rec,
            "opt_action_rec": opt_act_rec
        })

    save_results_plots(results_list, plot_title="Boltzmann Experiment Results On Various Tau Values",
                       results_folder='results', pdf_name='boltzmann_various_tau_values.pdf')