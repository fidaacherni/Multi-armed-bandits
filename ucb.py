from typing import Tuple, List
import numpy as np
from heroes import Heroes
from helpers import run_trials, save_results_plots


def ucb(
        heroes: Heroes,
        c: float,
        init_value: float = .0
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Perform Upper Confidence Bound (UCB) action selection for a bandit problem.

    :param heroes: A bandit problem, instantiated from the Heroes class.
    :param c: The exploration coefficient that balances exploration vs. exploitation.
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
    n_selected = [0] * num_heroes  # Number of times each hero has been selected
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

    for t in range(1, heroes.total_quests + 1):
        ######### WRITE YOUR CODE HERE
        ucb_values = []

        for i in range(num_heroes):
            if n_selected[i] == 0:
                # If hero has not been selected yet, set UCB to a very high value to encourage selection
                ucb_values.append(float('inf'))
            else:
                # Calculate the UCB value for each hero
                average_reward = values[i]
                confidence_bound = c * np.sqrt((2 * np.log(t)) / n_selected[i])
                ucb_values.append(average_reward + confidence_bound)

        # Select the hero with the maximum UCB value
        action = np.argmax(ucb_values)

        # The chosen hero attempts the quest
        reward = heroes.attempt_quest(action)
        rew_record.append(reward)

        # Update the selection count for the chosen hero
        n_selected[action] += 1

        # Update total rewards and calculate running average reward
        total_rewards += reward
        avg_ret_record.append(total_rewards / t)

        # Calculate regret: the difference between optimal and actual reward
        regret = optimal_reward - reward
        total_regret += regret
        tot_reg_record.append(total_regret)

        # Check if the chosen hero was the optimal one
        opt_action_record.append(1 if action == optimal_hero_index else 0)

        # Update the value estimates for the chosen hero using incremental average
        values[action] += (reward - values[action]) / n_selected[action]
        #########

    return rew_record, avg_ret_record, tot_reg_record, np.cumsum(opt_action_record) / (
        np.arange(1, heroes.total_quests + 1))


if __name__ == "__main__":
    # Define the bandit problem
    heroes = Heroes(total_quests=3000, true_probability_list=[0.35, 0.6, 0.1])

    # Test various c values
    c_values = [0.0, 0.5, 2.0]
    results_list = []
    for c in c_values:
        rew_rec, avg_ret_rec, tot_reg_rec, opt_act_rec = run_trials(30,
                                                                    heroes=heroes, bandit_method=ucb,
                                                                    c=c, init_value=0.0)

        results_list.append({
            'exp_name': f'c={c}',
            'reward_rec': rew_rec,
            'average_rew_rec': avg_ret_rec,
            'tot_reg_rec': tot_reg_rec,
            'opt_action_rec': opt_act_rec
        })

    save_results_plots(results_list, plot_title='UCB Experiment Results On Various C Values',
                       results_folder='results', pdf_name='ucb_various_c_values.pdf')
