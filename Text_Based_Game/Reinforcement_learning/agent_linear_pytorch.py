from string import punctuation, digits
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import framework as framework
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DEBUG = False

GAMMA = 0.5 # discounted factor
TRAINING_EP = 0.5 # epsilon-greedy parameter for training
TESTING_EP = 0.05 # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25 # number of episodes for training at each epoch
NUM_EPIS_TEST = 50 #number of episodes for testing
ALPHA = 0.001 # learning rate for training

actions = framework.get_actions()
objects = framework.get_objects()
NUM_ACTIONS = len(actions)
NUM_OBJECTS = len(objects)

model = None
optimizer = None

def tuple2index(action_index, object_index):
    return action_index * NUM_OBJECTS + object_index

def index2tuple(index):
    return index // NUM_OBJECTS, index % NUM_OBJECTS


# bag-of-words embedding
def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string descriptions
    Returns a dictionary of unique unigrams occurring over the input
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        #print(text[0])
        word_list = extract_words(text[0])
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vector(state_desc, dictionary):
    """
    Inputs a string state description
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words vector representation of the state
    The returned vector is of dimension m, where m the total number of entries in the dictionary.
    """
    state_vector = np.zeros([len(dictionary)])
    word_list = extract_words(state_desc)
    for word in word_list:
        if word in dictionary:
            state_vector[dictionary[word]] += 1

    # return state_vector
    return torch.FloatTensor(state_vector)


def epsilon_greedy(state_vector, epsilon):
    coin = np.random.random_sample()
    if (coin < epsilon):
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:
        with torch.no_grad():
            q_values = model(state_vector)
            index = torch.argmax(q_values)
            action_index, object_index = index2tuple(index)
    return (action_index, object_index)


def linear_Q_Learning(current_state_vector, action_index, object_index, reward, next_state_vector, terminal):

    with torch.no_grad():
        q_values_next = model(next_state_vector)
    maxq_next = torch.max(q_values_next)

    q_values = model(current_state_vector)
    q_value_cur = q_values[tuple2index(action_index, object_index)]

    target = reward + GAMMA * maxq_next * (1 - terminal)

    loss = (q_value_cur - target).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return(loss)


def run_episode(for_training):
    """
        Runs one episode
        If for training, update Q function
        If for testing, computes and return cumulative discounted reward
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    gamma_step = 1
    epi_reward = 0

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = extract_bow_feature_vector(current_state, dictionary)

        (action_index, object_index) = epsilon_greedy(current_state_vector, epsilon)
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(current_room_desc, current_quest_desc, action_index, object_index)

        if for_training:
            # update Q-function.
            next_state = next_room_desc + next_quest_desc
            next_state_vector = extract_bow_feature_vector(next_state, dictionary)
            linear_Q_Learning(current_state_vector, action_index, object_index, reward, next_state_vector, terminal)

        if not for_training:
            # update reward
            epi_reward = epi_reward + gamma_step * reward
            gamma_step = gamma_step * GAMMA

        # prepare next step
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc


    if not for_training:
        return epi_reward



def run_epoch():
    """
    Runs one epoch and returns reward averaged over test episodes
    """
    rewards = []

    for episode in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for episode in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """
    Returns array of test reward per epoch for one run
    """
    global model, optimizer
    model = nn.Linear(state_dim, action_dim)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)
    # optimizer = optim.Adam(model.parameters())

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for epoch in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description("Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
            np.mean(single_run_epoch_rewards_test),
            utils.ewma(single_run_epoch_rewards_test)
        ))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = bag_of_words(state_texts)
    state_dim = len(dictionary)
    action_dim = NUM_ACTIONS*NUM_OBJECTS

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = [] # shape NUM_RUNS * NUM_EPOCHS

    for nrun in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fix, ax = plt.subplots()
    ax.plot(x, np.mean(epoch_rewards_test,axis=0)) # plot reward per epoch averaged per run
    ax.set_xlabel('Epochs')
    ax.set_ylabel('reward')
    ax.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, ALPHA=%.4f' %(NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
