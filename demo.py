import numpy as np
import tensorflow as tf
import environment
import os


def start():
    env.reset(episode)
    present = [True for i in range(number)]
    observations = env.get_observation(present)
    poss_actions = env.possible_actions(present)
    #epsilon = max(1 - episode / 1110, 0.1)  # form 1 to 0.1 linearly till 1000 episodes
    epsilon = max(1 - episode / 275, 0.1)  # form 1 to 0.1 linearly till 250 episodes
    for step in range(steps):
        present = [True if env.agents[i][2] else False for i in range(number)]
        observations, poss_actions, done = play_one_step(observations, epsilon, poss_actions, step, present)
        if print == "map":
            env.render()
        if done:
            break


def play_one_step(states, epsilon, poss_actions, step, present):
    actions = list()
    for i in range(number):
        if present[i]:
            state = states[i]
            poss_act = poss_actions[i]
            model = models[i]
            action = epsilon_greedy_policy(state, epsilon, poss_act, model)
            actions.append(action)
        else:
            actions.append(None)

    next_state, next_poss_action, rewards, done, present = env.step(actions, present)

    if step == 999:
        done = True

    return next_state, next_poss_action, done


def epsilon_greedy_policy(state, epsilon, poss_act, model):
    if np.random.rand() < epsilon:
        return np.random.choice(poss_act)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)
        action = max_poss_action(Q_values[0], poss_act)
        return action


def max_poss_action(Q_values, poss_act):
    max_action = np.argmax(Q_values)
    if max_action not in poss_act:
        sorted = np.argsort(Q_values)
        n = -2
        action = sorted[n]
        while action not in poss_act:
            n -= 1
            action = sorted[n]
    else:
        action = max_action
    return action


def load_models():
    models = list()
    for i in range(number):
        checkpoint_path = "./{}/1- agent_{}/training_{}/model_{}/cp.ckpt".format(folder, number, episode, i)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_dir)
        models.append(model)
    return models


"""
To start the DEMO set the following variable:
-folder = the name of the folder were the models you want to load are saved
-number = number of agents in the system
-map = between "small"(for one agent), "open_2"(for 2 agents), "open_3"(for 3 agents), "open"(for 10 agents)
-print = "map" if you want to look at the whole map
         "observations" if you want to look at the observations in input to the model (of the first agent)
-steps = how many steps for episode should be showed (1000 for the fool episode)
-first_ep and last_ep = from which to which saved model you want to be showed 
"""
if __name__ == '__main__':
    folder = "NEW"
    number = 1
    map = "small"
    print = "map"  # or observations
    steps = 500
    first_ep = 2900
    last_ep = 2900

    for ep in range((first_ep//100), last_ep//100+1):
        episode = ep*100
        env = environment.Environment(map, number, print)
        models = load_models()
        start()


