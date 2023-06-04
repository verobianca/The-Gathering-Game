import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from tqdm import tqdm
import environment
import os
import pickle
from memory_profiler import profile


class DDQN:
    def __init__(self, folder, number=1, map="small", ep=0, print="none"):
        self.number = number
        self.n_outputs = 8  # ==env.action_space.n
        self.batch_size = 32
        self.discount_factor = 0.99
        self.optimizer = keras.optimizers.Adam(lr=1e-3, clipvalue=10.0)
        self.loss_fn = keras.losses.mean_squared_error
        self.C = 10000  # update step for target net
        self.update = 0
        self.print = print
        self.folder = folder
        self.env = environment.Environment(map, self.number, self.print)
        if ep == 0:
            self.models, self.target_models, self.replay_buffers = self.q_net()
        else:
            self.models, self.target_models, self.replay_buffers = self.load_models_buffer(ep)

    def q_net(self):
        models = list()
        target_models = list()
        replay_buffers = list()
        for i in range(self.number):
            model = keras.Sequential([
                keras.layers.Conv2D(filters=6, kernel_size=3, strides=2, activation="relu", input_shape=(20, 21, 1)),
                keras.layers.Conv2D(filters=6, kernel_size=3, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.n_outputs)
            ])

            target_model = keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())
            replay_buffer = deque(maxlen=100000)
            models.append(model)
            target_models.append(target_model)
            replay_buffers.append(replay_buffer)
        return models, target_models, replay_buffers

    def start(self, ep=0):
        for episode in tqdm(range(ep+1, 4001)):
            if episode % 100 == 0:
                self.save_models(episode)
            self.env.reset(episode)
            present = [True for i in range(self.number)]
            observations = self.env.get_observation(present)
            poss_actions = self.env.possible_actions(present)
            epsilon = max(1 - episode / 1110, 0.1)  # form 1 to 0.1 linearly till 1000 episodes
            for step in range(1000):
                present = [True if self.env.agents[i][2] else False for i in range(self.number)]
                observations, poss_actions, done = self.play_one_step(observations, epsilon, poss_actions, step, present)
                if self.print == "map":
                    self.env.render()

                if done:
                    break
                if episode > 5:
                    self.training_step()

    def play_one_step(self, states, epsilon, poss_actions, step, present):
        actions = list()
        for i in range(self.number):
            if present[i]:
                state = states[i]
                poss_act = poss_actions[i]
                model = self.models[i]
                action = self.epsilon_greedy_policy(state, epsilon, poss_act, model)
                actions.append(action)
            else:
                actions.append(None)

        next_state, next_poss_action, rewards, done, present = self.env.step(actions, present)

        if step == 999:
            done = True

        self.add_to_replay_buffer(states, actions, rewards, next_state, done, next_poss_action, present)
        return next_state, next_poss_action, done

    def epsilon_greedy_policy(self, state, epsilon, poss_act, model):
        if np.random.rand() < epsilon:
            return np.random.choice(poss_act)
        else:
            Q_values = model.predict(state[np.newaxis], verbose=0)
            action = self.max_poss_action(Q_values[0], poss_act)
            return action

    def max_poss_action(self, Q_values, poss_act):
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

    def add_to_replay_buffer(self, states, actions, rewards, next_state, done, next_poss_action, present):
        for i in range(self.number):
            if present[i]:
                self.replay_buffers[i].append([states[i], actions[i], rewards[i], next_state[i], next_poss_action[i], done])

    @profile
    def training_step(self):
        for i in range(self.number):
            model = self.models[i]
            target_model = self.target_models[i]
            batch = self.sample_experience(i)
            states, actions, rewards, next_states, next_poss_actions, dones = [
                np.array([experience[field_index] for experience in batch]) for field_index in range(6)]

            next_Q_values = model(next_states)
            max_next_action = [self.max_poss_action(next_Q_values[i], next_poss_actions[i]) for i in range(self.batch_size)]
            next_Q_values_target = target_model.predict(next_states, verbose=0)
            mask_target = tf.one_hot(max_next_action, self.n_outputs)
            max_next_Q_values = tf.reduce_sum(next_Q_values_target * mask_target, axis=1, keepdims=True)
            target_Q_values = (rewards + (1 - dones) * self.discount_factor * max_next_Q_values)
            mask = tf.one_hot(actions, self.n_outputs)
            with tf.GradientTape() as tape:
                all_Q_values = model(states)
                Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
                loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        self.update += 1
        if self.update == self.C:
            for i in range(self.number):
                self.update_target_net(self.models[i], self.target_models[i])

    def sample_experience(self, i):
        indices = np.random.randint(len(self.replay_buffers[i]), size=self.batch_size)
        batch = [self.replay_buffers[i][index] for index in indices]
        return batch

    def update_target_net(self, model, target_model):
        target_model.set_weights(model.get_weights())

    def save_models(self, episode):
        for i in range(self.number):
            checkpoint_path = "./{}/agent_{}/training_{}/model_{}/cp.ckpt".format(self.folder, self.number, episode, i)
            checkpoint_dir = os.path.dirname(checkpoint_path)
            self.models[i].save(checkpoint_dir)

        buffer = open(r'./{}/agent_{}/training_{}/buffer.pkl'.format(self.folder, self.number, episode), 'wb')
        pickle.dump(self.replay_buffers, buffer)
        buffer.close()

    def load_models_buffer(self, episode):
        models = list()
        target_models = list()

        for i in range(self.number):
            checkpoint_path = "./{}/agent_{}/training_{}/model_{}/cp.ckpt".format(self.folder, self.number, episode, i)
            checkpoint_dir = os.path.dirname(checkpoint_path)
            model = tf.keras.models.load_model(checkpoint_dir)
            models.append(model)
            target_model = keras.models.clone_model(model)
            self.update_target_net(model, target_model)
            target_models.append(target_model)

        file = open(r'./{}/agent_{}/training_{}/buffer.pkl'.format(self.folder, self.number, episode), 'rb')
        buffer = pickle.load(file)
        file.close()
        return models, target_models, buffer


