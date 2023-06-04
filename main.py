import DDQN
"""
To start the TRAINING set the following variable:
-folder = the name of the folder you want to save the models/the folder were the models you want to load are saved
-number = number of agents in the system
-map = between "small"(for one agent), "open_2"(for 2 agents), "open_3"(for 3 agents), "open"(for 10 agents)
-ep = the episode from which you want to start the training (0 if you want to train a new model)
-print = "none" if you just want to train it
         "map" if you want to look at the whole map while training
         "observations" if you want to look at the observations in input to the model (of the first agent) while training 
"""
if __name__ == '__main__':
    folder = "best_one"
    number = 2
    map = "open"
    ep = 0
    print = "none"

    ddqn = DDQN.DDQN(folder, number, map, ep, print)
    ddqn.start(ep)
