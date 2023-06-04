import numpy as np
import matplotlib.pyplot as plt
import cv2

# empty black[0, 0, 0]
# apples green [0, 255, 0]
# agent red [255, 0, 0]
# other agents blu [0, 0, 255]
# tag yellow[150, 150, 0]
# wall gray [100, 100, 100]


class Environment:
    def __init__(self, map="small", number=1, print="map"):
        self.number = number
        self.map = map
        self.set_map()
        self.set_agents()
        self.print = print
        if self.print == "map":
            self.fig, self.ax = plt.subplots(1, 1)
            self.myplot = self.ax.imshow(self.grid.astype('uint8'))
        elif self.print == "observations":
            window = np.zeros((20, 21))
            window[0, 0] = 255
            self.fig, self.ax = plt.subplots(1, 1)
            self.myplot = self.ax.imshow(window.astype('uint8'))

    def set_map(self):
        if self.map == "small":
            # set size and positions
            self.size = (9, 26, 3)
            self.grid = np.zeros(self.size)

            # walls
            self.grid[0, :] = [100, 100, 100]
            self.grid[-1, :] = [100, 100, 100]
            self.grid[:, 0] = [100, 100, 100]
            self.grid[:, -1] = [100, 100, 100]

            # apples
            self.apples = {(2, 3): 1, (2, 8): 1, (2, 13): 1,  (2, 18): 1, (2, 23): 1, (6, 5): 1,
                           (6, 10): 1, (6, 15): 1, (6, 20): 1}

            keys = list(self.apples.keys())
            for a in keys:
                i, j = a
                self.apples[i+1, j] = 1
                self.apples[i, j+1] = 1
                self.apples[i - 1, j] = 1
                self.apples[i, j-1] = 1

            for i in self.apples:
                self.grid[i][1] = 255

        elif self.map == "open_2":
            # set size and positions
            self.size = (14, 19, 3)
            self.grid = np.zeros(self.size)

            # walls
            self.grid[0, :] = [100, 100, 100]
            self.grid[-1, :] = [100, 100, 100]
            self.grid[:, 0] = [100, 100, 100]
            self.grid[:, -1] = [100, 100, 100]

            # apples
            self.apples = {(2, 3): 1, (3, 7): 1, (3, 12): 1, (5, 15): 1, (6, 12): 1,
                           (7, 3): 1, (7, 8): 1, (9, 6): 1, (10, 13): 1, (10, 2): 1}

            keys = list(self.apples.keys())
            for a in keys:
                i, j = a
                self.apples[i + 1, j] = 1
                self.apples[i, j + 1] = 1
                self.apples[i - 1, j] = 1
                self.apples[i, j - 1] = 1

            for i in self.apples:
                self.grid[i][1] = 255
        elif self.map == "open_3":
            # set size and positions
            self.size = (15, 23, 3)
            self.grid = np.zeros(self.size)

            # walls
            self.grid[0, :] = [100, 100, 100]
            self.grid[-1, :] = [100, 100, 100]
            self.grid[:, 0] = [100, 100, 100]
            self.grid[:, -1] = [100, 100, 100]

            # apples
            self.apples = {(2, 3): 1, (2, 16): 1, (3, 7): 1, (3, 12): 1, (3, 20): 1,  (5, 15): 1, (6, 12): 1,
                           (7, 3): 1, (7, 8): 1, (8, 18): 1, (9, 6): 1, (10, 13): 1, (10, 2): 1, (12, 7): 1}

            keys = list(self.apples.keys())
            for a in keys:
                i, j = a
                self.apples[i+1, j] = 1
                self.apples[i, j+1] = 1
                self.apples[i - 1, j] = 1
                self.apples[i, j-1] = 1

            for i in self.apples:
                self.grid[i][1] = 255

        elif self.map == "open":
            # set size and positions
            self.size = (15, 30, 3)  # row, column
            self.grid = np.zeros(self.size)

            # walls
            self.grid[0, :] = [100, 100, 100]
            self.grid[-1, :] = [100, 100, 100]
            self.grid[:, 0] = [100, 100, 100]
            self.grid[:, -1] = [100, 100, 100]

            # apples
            self.apples = {(2, 3): 1, (2, 15): 1, (2, 26): 1, (3, 8):1, (3, 12): 1, (3, 19): 1, (3, 23): 1,  (4, 5): 1,
                           (5, 15): 1, (6, 12): 1, (6, 26): 1, (7, 3): 1, (7, 8): 1, (7, 23): 1, (8, 17): 1, (9, 6): 1,
                           (9, 12): 1, (10, 2): 1, (10, 22): 1, (10, 25): 1, (12, 7): 1, (12, 18): 1}

            keys = list(self.apples.keys())
            for a in keys:
                i, j = a
                self.apples[i+1, j] = 1
                self.apples[i, j+1] = 1
                self.apples[i - 1, j] = 1
                self.apples[i, j-1] = 1

            for i in self.apples:
                self.grid[i][1] = 255

    def set_agents(self):
        agents = list()
        for i in range(self.number):
            agents.append([(self.size[0]-2, self.size[1]-2-(i*2)), 0, 1, 0])  # pos, orientation, presence, step counter after tagged
            self.grid[self.size[0] - 2, self.size[1] - 2 - (i * 2)] = [255, 0, 0]
        self.agents = np.array(agents, dtype=object)

    def render(self):
        self.myplot.set_data(self.grid.astype('uint8'))
        self.fig.canvas.draw_idle()
        plt.pause(0.000001)
        #plt.pause(0.5)

    def reset(self, n):
        self.set_map()
        self.set_agents()
        if self.print != "none":
            self.ax.title.set_text('Episode {}'.format(n))

    def get_observation(self, present):
        observations = list()

        for i in range(self.number):
            if present[i]:
                pos = self.agents[i][0]
                orientation = self.agents[i][1]
                grid_copy = self.grid.copy()
                grid_copy[(pos[0], pos[1])] = [0, 0, 255]
                grid_copy = cv2.cvtColor(np.float32(grid_copy), cv2.COLOR_RGB2GRAY)
                pad = 21
                grid_copy = np.pad(grid_copy, pad)

                pos_0 = pos[0] + pad
                pos_1 = pos[1] + pad

                if orientation == 0:  # up
                    window = grid_copy[pos_0 - 19: pos_0+1, pos_1 - 10: pos_1 + 10 + 1].copy()
                elif orientation == 1:  # right
                    window = grid_copy[pos_0 - 10: pos_0 + 10 + 1, pos_1: pos_1 + 20].copy()
                    window = np.rot90(window)
                elif orientation == 2:  # down
                    window = grid_copy[pos_0: pos_0 + 20, pos_1 - 10: pos_1 + 10 + 1].copy()
                    window = np.rot90(window, 2)
                else: # left
                    window = grid_copy[pos_0 - 10: pos_0 + 10 + 1, pos_1 - 19: pos_1+1].copy()
                    window = np.rot90(window, 3)

                if self.print == "observations" and i == 0:
                    self.print_observation(window)

                observations.append(window/255)
            else:
                observations.append(None)
        return observations

    def print_observation(self, window):
        self.myplot.set_data(window.astype('uint8'))
        self.fig.canvas.draw_idle()
        plt.pause(0.000001)

    # check which are the legal action in a given state
    def possible_actions(self, present):
        # 0 step forward
        # 1 step right
        # 2 step backward
        # 3 step left
        # 4 rotate right
        # 5 rotate left
        # 6 tag
        # 7 stay still

        poss_actions = list()
        for i in range(self.number):
            if present[i]:
                poss_action = [4, 5, 6, 7]
                pos = self.agents[i][0]
                orientation = self.agents[i][1]
                """
                # up
                if (self.grid[pos[0] - 1, pos[1]] != [100, 100, 100]).all():
                    if orientation in [0, 1]:
                        poss_action.append(orientation)
                    else:
                        poss_action.append((orientation + 2) % 4)
                # right
                if (self.grid[pos[0], pos[1] + 1] != [100, 100, 100]).all():
                    if orientation in [0, 2]:
                        poss_action.append(orientation + 1)
                    else:
                        poss_action.append(orientation - 1)
                # down
                if (self.grid[pos[0] + 1, pos[1]] != [100, 100, 100]).all():
                    if orientation in [0, 2]:
                        poss_action.append((orientation + 2) % 4)
                    else:
                        poss_action.append(orientation)
                # left
                if (self.grid[pos[0], pos[1] - 1] != [100, 100, 100]).all():
                    if orientation == [0, 2]:
                        poss_action.append((orientation + 3) % 4)
                    else:
                        poss_action.append((orientation + 1) % 4)
                """
                # up
                if (self.grid[pos[0] - 1, pos[1]] != [100, 100, 100]).all():
                    if orientation == 0:
                        poss_action.append(0)
                    elif orientation == 1:
                        poss_action.append(3)
                    elif orientation == 2:
                        poss_action.append(2)
                    elif orientation == 3:
                        poss_action.append(1)
                # right
                if (self.grid[pos[0], pos[1] + 1] != [100, 100, 100]).all():
                    if orientation == 0:
                        poss_action.append(1)
                    elif orientation == 1:
                        poss_action.append(0)
                    elif orientation == 2:
                        poss_action.append(3)
                    elif orientation == 3:
                        poss_action.append(2)
                # down
                if (self.grid[pos[0] + 1, pos[1]] != [100, 100, 100]).all():
                    if orientation == 0:
                        poss_action.append(2)
                    elif orientation == 1:
                        poss_action.append(1)
                    elif orientation == 2:
                        poss_action.append(0)
                    elif orientation == 3:
                        poss_action.append(3)
                # left
                if (self.grid[pos[0], pos[1] - 1] != [100, 100, 100]).all():
                    if orientation == 0:
                        poss_action.append(3)
                    elif orientation == 1:
                        poss_action.append(2)
                    elif orientation == 2:
                        poss_action.append(1)
                    elif orientation == 3:
                        poss_action.append(0)
                poss_actions.append(poss_action)
            else:
                poss_actions.append(None)
        return poss_actions

    # do a full step for each agent, get the rewards, change the environment accordingly and get the new states
    def step(self, actions, present):
        # delete yellow tag
        self.grid = np.array(
            [[[0, 0, 0] if (self.grid[j, i] == [150, 150, 0]).all() else self.grid[j, i]
              for i in range(self.grid.shape[1])] for j in range(self.grid.shape[0])])

        done = False
        rewards = list()
        present_copy = present.copy()
        for i in range(self.number):
            if present[i]:
                reward = 0
                pos = self.agents[i][0]
                orientation = self.agents[i][1]
                action = actions[i]

                if action == 0:
                    if orientation == 0:
                        new_pos = (pos[0] - 1, pos[1])
                    elif orientation == 1:
                        new_pos = (pos[0], pos[1] + 1)
                    elif orientation == 2:
                        new_pos = (pos[0] + 1, pos[1])
                    else:
                        new_pos = (pos[0], pos[1] - 1)

                    self.agents[i][0] = new_pos
                    self.grid[pos] = [0, 0, 0]
                    if (self.grid[new_pos] == [0, 255, 0]).all():
                        reward = 1
                        self.apples[new_pos] = 0
                    self.grid[new_pos] = [255, 0, 0]

                elif action == 1:
                    if orientation == 0:
                        new_pos = (pos[0], pos[1] + 1)
                    elif orientation == 1:
                        new_pos = (pos[0] + 1, pos[1])
                    elif orientation == 2:
                        new_pos = (pos[0], pos[1] - 1)
                    else:
                        new_pos = (pos[0] - 1, pos[1])

                    self.agents[i][0] = new_pos
                    self.grid[pos] = [0, 0, 0]
                    if (self.grid[new_pos] == [0, 255, 0]).all():
                        reward = 1
                        self.apples[new_pos] = 0
                    self.grid[new_pos] = [255, 0, 0]

                elif action == 2:
                    if orientation == 0:
                        new_pos = (pos[0] + 1, pos[1])
                    elif orientation == 1:
                        new_pos = (pos[0], pos[1] - 1)
                    elif orientation == 2:
                        new_pos = (pos[0] - 1, pos[1])
                    else:
                        new_pos = (pos[0], pos[1] + 1)

                    self.agents[i][0] = new_pos
                    self.grid[pos] = [0, 0, 0]
                    if (self.grid[new_pos] == [0, 255, 0]).all():
                        reward = 1
                        self.apples[new_pos] = 0
                    self.grid[new_pos] = [255, 0, 0]

                elif action == 3:
                    if orientation == 0:
                        new_pos = (pos[0], pos[1] - 1)
                    elif orientation == 1:
                        new_pos = (pos[0] - 1, pos[1])
                    elif orientation == 2:
                        new_pos = (pos[0], pos[1] + 1)
                    else:
                        new_pos = (pos[0] + 1, pos[1])

                    self.agents[i][0] = new_pos
                    self.grid[pos] = [0, 0, 0]
                    if (self.grid[new_pos] == [0, 255, 0]).all():
                        reward = 1
                        self.apples[new_pos] = 0
                    self.grid[new_pos] = [255, 0, 0]

                elif action == 4:
                    self.agents[i][1] = (self.agents[i][1] + 1) % 4
                    if (self.grid[self.agents[i][0]] == [0, 255, 0]).all():
                        reward = 1
                        self.grid[self.agents[i][0]] = [255, 0, 0]
                        self.apples[self.agents[i][0]] = 0

                elif action == 5:
                    self.agents[i][1] = (self.agents[i][1] + 3) % 4
                    if (self.grid[self.agents[i][0]] == [0, 255, 0]).all():
                        reward = 1
                        self.grid[self.agents[i][0]] = [255, 0, 0]
                        self.apples[self.agents[i][0]] = 0

                elif action == 6:
                    if (self.grid[self.agents[i][0]] == [0, 255, 0]).all():
                        reward = 1
                        self.grid[self.agents[i][0]] = [255, 0, 0]
                        self.apples[self.agents[i][0]] = 0

                    if orientation == 0:  # up
                        j1, j2, i1, i2 = max(0, pos[0] - 20), pos[0], max(0, pos[1] - 2), min(self.size[1], pos[1] + 3)
                    elif orientation == 1:  # right
                        j1, j2, i1, i2 = max(0, pos[0] - 2), min(self.size[0], pos[0] + 3), max(0, pos[1]+1), min(self.size[1], pos[1] + 21)
                    elif orientation == 2:  # down
                        j1, j2, i1, i2 = max(0, pos[0]+1), min(self.size[0], pos[0] + 21), max(0, pos[1] - 2), min(self.size[1], pos[1] + 3)
                    else:  # left
                        j1, j2, i1, i2 = max(0, pos[0] - 2), min(self.size[0], pos[0] + 3), max(0, pos[1] - 20), pos[1]

                    for jj in range(j1, j2):
                        for ii in range(i1, i2):
                            # remove tagged agents
                            if (self.grid[jj, ii] == [255, 0, 0]).all():
                                self.grid[jj, ii] = [0, 0, 0]
                                for n in [z for z in range(len(self.agents)) if self.agents[z][0] == (jj, ii)]:
                                    self.agents[n][2] = 0
                                    present[n] = False
                                    present_copy[n] = False
                            # tag yellow
                            if (self.grid[jj, ii] == [0, 0, 0]).all():
                                self.grid[jj, ii] = [150, 150, 0]

                elif action == 7:
                    if (self.grid[self.agents[i][0]] == [0, 255, 0]).all():
                        reward = 1
                        self.grid[self.agents[i][0]] = [255, 0, 0]
                        self.apples[self.agents[i][0]] = 0

                rewards.append(reward)

            else:
                rewards.append(0)
                self.agents[i][3] += 1
                if self.agents[i][3] == 25:
                    self.agents[i][3] = 0
                    self.agents[i][2] = 1
                    present_copy[i] = True

        self.check_apples()

        # if there aren't any apples left stop the episode
        if 1 not in list(self.apples.values()):
            done = True

        next_states = self.get_observation(present_copy)
        poss_next_actions = self.possible_actions(present_copy)
        return next_states, poss_next_actions, rewards, done, present

    # check if apple can respawn
    def check_apples(self):
        for key in [k for k, v in self.apples.items() if v == 0]:
            n = 0
            for j in range(key[0] - 2, key[0] + 3):
                for i in range(key[1] - 2, key[1] + 3):
                    try:
                        if (self.grid[j, i] == [0, 255, 0]).all():
                            n += 1
                    except:
                        pass
            if n == 0:
                p = 0
            elif n in [1, 2]:
                p = 0.01
            elif n in [3, 4]:
                p = 0.05
            else:
                p = 0.1
            if np.random.rand() < p:
                self.apples[key] = 1
                self.grid[key] = [0, 255, 0]
