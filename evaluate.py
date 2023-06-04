import numpy as np
import tensorflow as tf
import environment
import os
import matplotlib.pyplot as plt


def efficiency(rewards):
    return np.average(np.sum(rewards, axis=1))


def equality(rewards):
    ss = 0
    for i in rewards:
        for j in rewards:
            ss += np.abs(np.sum(i) - np.sum(j))

    return 1 - np.divide(ss, 2*rewards.shape[0]*rewards.sum())


def sustainability(rewards):
    t = list()
    for i in rewards:
        ti = np.sum([t for t in range(len(i)) if i[t] == 1])
        t.append(ti)
    return np.average(t)


def peace(present):
    return np.divide(present.shape[0] * present.shape[1] - present.sum(), present.shape[0])


def evaluate(all_rewards, all_present):
    rewards = list()

    for i in range(all_rewards.shape[0]):
        rewards.append(all_rewards[i])
    rewards = np.array(rewards)
    rewards = rewards.transpose()
    all_present = all_present.transpose()

    U = efficiency(rewards)
    if number != 1:
        E = equality(rewards)
        S = sustainability(rewards)
        P = peace(all_present)
    else:
        E = 0
        S = 0
        P = 0
    return U, E, S, P


def start():
    all_rewards = list()
    all_present = list()
    env.reset(episode)
    present = [True for i in range(number)]
    observations = env.get_observation(present)
    poss_actions = env.possible_actions(present)
    #epsilon = max(1 - episode / 1110, 0.1)  # form 1 to 0.1 linearly till 1000 episodes
    epsilon = max(1 - episode / 275, 0.1)  # form 1 to 0.1 linearly till 250 episodes
    for step in range(1000):
        present = [True if env.agents[i][2] else False for i in range(number)]
        observations, poss_actions, rewards, done = play_one_step(observations, epsilon, poss_actions, step, present)
        all_rewards.append(rewards)
        all_present.append(present)
        if done:
            break

    print(np.sum(all_rewards, axis=0))
    U, E, S, P = evaluate(np.array(all_rewards), np.array(all_present))
    return U, E, S, P


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

    return next_state, next_poss_action, rewards, done


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
        checkpoint_path = "./{}/agent_{}/training_{}/model_{}/cp.ckpt".format(folder,number, episode, i)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_dir)
        models.append(model)
    return models


def load():
    env = environment.Environment(map, number)
    models = load_models()
    return env, models


"""
To start the EVALUATION set the following variable:
-folder = the name of the folder were the models you want to load are saved
-number = number of agents in the system
-map = between "small"(for one agent), "open_2"(for 2 agents), "open_3"(for 3 agents), "open"(for 10 agents)
-first_ep and last_ep = from which to which saved model you want to do the evaluation
"""
if __name__ == '__main__':
    folder = "NEW"
    number = 2
    map = "open_2"
    first_ep = 100
    last_ep = 1000

    U, E, S, P = list(), list(), list(), list()
    for ep in range((first_ep//100), last_ep//100+1):
        episode = ep*100
        print('Episode {}'.format(episode))
        env = environment.Environment(map, number, "none")
        models = load_models()
        u, e, s, p = start()

        U.append(u)
        if number != 1:
            E.append(e)
            S.append(s)
            P.append(p)

    if number == 1:
        figure, axis = plt.subplots(1, 1)
        axis.plot(U)
        axis.set_ylim(bottom=0, top=400)
        axis.set_title('Efficiency')
    else:
        figure, axis = plt.subplots(2, 2)
        axis[0, 0].plot(U)
        axis[0, 0].set_ylim(bottom=0, top=400)
        axis[0, 0].set_title('Efficiency')

        axis[0, 1].plot(E)
        axis[0, 1].set_title('Equality')

        axis[1, 0].plot(E)
        axis[1, 0].set_title('Sustainability')

        axis[1, 1].plot(P)
        axis[1, 1].set_title('Peace')

    plt.show()
