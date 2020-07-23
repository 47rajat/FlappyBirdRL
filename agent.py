import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, Lambda, Add, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import VarianceScaling
from collections import deque
import matplotlib.pyplot as plt
import random
import os
import time
import math
import pickle

plt.style.use('ggplot')

random.seed(123)
np.random.seed(123)

class FlappyBirdRLAgent:
    """
    RL Agent that for playing Flappy Bird Game.
    """
    def __init__(self, model_name=None, actions=None, state_space=None, epsilon=1.0, eval_model=None):
        """
        Initializes the agent with the provided state_space and actions in either train mode or eval
        mode.

        Args
        ----
        `model_name` (str): Used to create a directory that saves the agent state and history
                            information that can be used to resume training from the last 
                            checkpoint.
        
        `actions` (int): The total number of actions agent can perform in the environment.

        `state_space` (tuple): Basically the dimension of the state space. Used as input for the
                                agent neural network.

        `epsilon` (float): [0, 1] that controls the agents probability of picking the random
                            action while using epsilon-greedy policy.
        
        `eval_model` (str): Path to model weights that is to be evaluated.
        
        Raises
        ------
        `ValueError`: If state_space or actions are not provided. If the value of epsilon is not
                        in the range [0, 1].
        """
        if state_space is None:
            raise ValueError('Error state space is not provided')

        if actions is None:
            raise ValueError('Error number of actions not provided')
            
        if epsilon > 1.0 or epsilon < 0:
            raise ValueError(f'Error epsilon value should be in [0, 1], provided value = {epsilon}')
        
        self._STATE_SPACE = state_space
        self._NUM_ACTIONS = actions
        
        if eval_model is not None:
            self._eval_mode = True
            self._init_eval_mode(eval_model=eval_model)
        else:
            self._eval_mode = False
            self._init_train_mode(model_name=model_name, epsilon=epsilon)
    

    def _init_eval_mode(self, eval_model=None):
        """
        Initializes the agent in evaluation mode.

        Args
        ----
        `eval_model` (str): Path to model weights that is to be evaluated.
        """
        self._init_models()
        self._prediction_model.load_weights(eval_model)

    def _init_train_mode(self, model_name=None, epsilon=1.0):
        """
        Initializes model in training mode.

        Args
        ----
        `model_name` (str): Used to create a directory that saves the agent state and history
                            information that can be used to resume training from the last 
                            checkpoint.

        `epsilon` (float): [0, 1] that controls the agents probability of picking the random
                            action while using epsilon-greedy policy.
        """

        # Number of state transition that are stored while training.
        self._MAX_REPLAY_BUFFER_SIZE = 100_000

        # Minimum number of state transitions that are collected before starting training. 
        self._MIN_REPLAY_BUFFER_SIZE = 1_000 

        self._BATCH_SIZE = 32
        
        # Discount factor
        self._GAMMA = 0.99 
        

        self._MIN_EPSILON = 1e-4
        self._EPSILON_DECAY_STEPS = 1e4
        self._EPSILON_DECAY_RATE = 10**(math.log10(self._MIN_EPSILON/epsilon)/self._EPSILON_DECAY_STEPS)
        
        # Minimum score above which an evaluation model is saved.
        self._SAVE_SCORE_THRESHOLD = 200
        
        # indicates whether we will store history/checkpoints for agent.
        self._MODEL_NAME = model_name 
        if self._MODEL_NAME is not None:
            self._MODEL_DIR = f'./models/{self._MODEL_NAME}'
            self._REPLAY_BUFFER_FILE = f'{self._MODEL_DIR}/replay_buffer.pkl'
            self._CHECKPOINT_FILE = f'{self._MODEL_DIR}/checkpoint.csv'
            self._HISTORY_FILE = f'{self._MODEL_DIR}/history.csv'
            self._HISTORY_HEADERS = ['episode', 'step', 'train_score', 'eval_score', 'best_score', 'epsilon', 'loss']
            # add fields for q values predicted by the network.
            for action in range(self._NUM_ACTIONS):
                self._HISTORY_HEADERS.append(f'q_avg_{action}')
                self._HISTORY_HEADERS.append(f'q_min_{action}')
                self._HISTORY_HEADERS.append(f'q_max_{action}')
            self._HISTORY_BUFFER_SIZE = 100 # used to smooth training score values.
            self._PLOT_DPI = 240 # dpi used while saving the plots.
            self._PROGRESS_FILE = f'{self._MODEL_DIR}/progress.png'
            self._LOSS_FILE = f'{self._MODEL_DIR}/loss.png'
            self._Q_VALUES_FILE = f'{self._MODEL_DIR}/q_values.png'
            self._ARCHITECTURE_FILE = f'{self._MODEL_DIR}/architecture.txt'
            self._PREDICTION_MODEL_WEIGHTS = f'{self._MODEL_DIR}/prediction_model_weights.h5'
            self._TARGET_MODEL_WEIGHTS = f'{self._MODEL_DIR}/target_model_weights.h5'

            # buffer to store action values predicted by the network in each episode
            self._q_values_history = [[] for action in range(self._NUM_ACTIONS)]

        
        self._train_score = 0
        self._eval_score = -5.0
        self._epsilon = epsilon
        self._replay_memory = deque(maxlen=self._MAX_REPLAY_BUFFER_SIZE)

        # Step 1: initialize all the models.
        self._init_models()
        # Step 2: initialize checkpoints file.
        self._init_checkpoints()

    def _init_models(self):
        """
        Initializes the prediction and the target model using random weights.
        """
        # prediction model is the one which will be trained on each step.
        self._prediction_model = self._build_model()
        # target model will be used to provide target Q values for prediction model to predict against.
        # Helps in stabilizing prediction model training.
        self._target_model = self._build_model()
        self._target_model.set_weights(self._prediction_model.get_weights())

    
    def _init_checkpoints(self):
        """
        Initializes the model from last checkpoint if a model name is provided during init. If the directory
        corresponding the the model name is not present, initializes model with zero values to start fresh
        training.

        If model name is not present initializes the _start_episode, _start_step and _best_score for the model.
        """
        if self._MODEL_NAME is None:
            self._start_episode, self._start_step, self._best_score = 0, 0, -5.0 # (least score)
            return
        
        # check if checkpoints for the provided model exists.
        if os.path.isdir(self._MODEL_DIR):
            with open(self._CHECKPOINT_FILE, 'r') as file:
                cnt = 0
                for line in file:
                    start_episode, start_step, best_score, epsilon = line.split(',')
                    cnt += 1
                    assert(cnt < 2)
            self._start_episode, self._start_step, self._best_score, self._epsilon = \
                        int(start_episode), int(start_step), float(best_score), float(epsilon[:-1])

            # For prediction model entire model is saved including optimizer states, so as to resume
            # training from the exact last state.
            self._prediction_model = load_model(self._PREDICTION_MODEL_WEIGHTS)
            
            # For target model just the model weights are saved.
            self._target_model.load_weights(self._TARGET_MODEL_WEIGHTS)

            with open(self._REPLAY_BUFFER_FILE, 'rb') as f:
                self._replay_memory = pickle.load(f)

            print(f'Resuming training for model={self._MODEL_NAME} with best_score = {self._best_score}',
                  f'from episode = {self._start_episode}, step = {self._start_step},',
                  f'replay_buffer_size = {len(self._replay_memory)} and epsilon = {self._epsilon}')
        
        else: # if not initialize everything with zero values.
            os.makedirs(self._MODEL_DIR)
            self._start_episode = 0
            self._start_step = 0
            self._best_score = -5.0 # least possible score.
            with open(self._CHECKPOINT_FILE, 'w') as f:
                f.write(f'{self._start_episode},{self._start_step},{self._best_score},{self._epsilon}\n')
            
            with open(self._HISTORY_FILE, 'w') as f:
                f.write(','.join(self._HISTORY_HEADERS))
                f.write('\n')
            
            with open(self._ARCHITECTURE_FILE, 'w') as f:
                f.write(f'STATE_SPACE = {self._STATE_SPACE}\n')
                f.write(f'NUM_ACTIONS = {self._NUM_ACTIONS}\n')
                f.write(f'MAX_REPLAY_BUFFER_SIZE = {self._MAX_REPLAY_BUFFER_SIZE}\n')
                f.write(f'MIN_REPLAY_BUFFER_SIZE = {self._MIN_REPLAY_BUFFER_SIZE}\n')
                f.write(f'BATCH_SIZE = {self._BATCH_SIZE}\n')
                f.write(f'GAMMA = {self._GAMMA}\n')
                f.write(f'EPSILON_DECAY_RATE = {self._EPSILON_DECAY_RATE}\n')
                f.write(f'MIN_EPSILON = {self._MIN_EPSILON}\n')
                f.write(f'SAVE_SCORE_THRESHOLD = {self._SAVE_SCORE_THRESHOLD}\n\n')
                f.write(self._prediction_model.to_json(indent=4))
            
            self._save_models_and_buffer(0)

            print(f'Starting training for new model = {self._MODEL_NAME}')

    def get_start_checkpoint(self):
        """
        Provide the episode and step number form which to resume (or in some cases start) training.

        Returns
        -------
        Integer tuple (`start_episode`, `start_step`).
        """
        return self._start_episode, self._start_step
    
    def get_action(self, state):
        """
        Provides the epsilon-greedy action that the agent will take.

        Args
        ----
        `state` (n-d numpy array): The current state in which agent has to pick an action.
        
        Returns
        -------
        An integer in the range `[0, self._NUM_ACTIONS).
        """

        # Decay epsilon value.
        if self._epsilon > self._MIN_EPSILON:
            self._epsilon *= self._EPSILON_DECAY_RATE
        
        random_action = np.random.randint(self._NUM_ACTIONS)
        best_action = self.get_greedy_action(state)

        return random_action if np.random.random() < self._epsilon else best_action

    def get_greedy_action(self, state):
        """
        Provides the greedy action(action with the maximum Q value) picked by the agent for the provided state.

        Args
        ----
        `state`(n-d numpy array): The current state in which the agent has to pick an action.
        
        Returns
        -------
        An integer in the range `[0, _NUM_ACTIONS)
        """
        return np.argmax(self.get_q_values(state))
    
    def get_q_values(self, state):
        '''
        Provides the Q-values for the provided state.

        Args
        ----
        `state` (n-d numpy array): The current un-normalized(image with pixel values between 0 and 255) state images.
        
        Returns
        -------
        List of floats of length `_NUM_ACTIONS` where each entry is the Q value for the provided state
        and the action represented by that index.
        '''
        q_values = self._prediction_model.predict(np.expand_dims(state, 0)/255.0)[0]
        if self._eval_mode == False and self._MODEL_NAME is not None:
            for action in range(self._NUM_ACTIONS):
                self._q_values_history[action].append(q_values[action])
        return q_values

    def get_train_score(self):
        """
        Provides the current training score of the agent.

        Returns
        -------
        An integer.
        """
        return self._train_score

    def get_eval_score(self):
        """
        Provides the current evaluation score of the agent.

        Returns
        -------
        An integer.
        """
        return self._eval_score
    
    def get_best_score(self):
        """
        Provides the highest score the agent has been able to achieve till the moment.

        Returns
        -------
        An integer.
        """
        return self._best_score

    def set_train_score(self, new_score):
        """
        Updates the value of `_train_score` with the provided value.

        Args
        ----
        `new_score` (int): An integer representing the new value to which `_train_score` should be updated to.
        """
        self._train_score = new_score
    
    def set_eval_score(self, new_score):
        """
        Updates the value of `_eval_score` with the provided value.

        Args
        ----
        `new_score` (int): An integer representing the new value to which `_eval_score` should be updated to.
        """
        self._eval_score = new_score

    def _build_model(self):
        """
        Builds a CNN model that will be used by the agent to predict Q-values.

        Returns
        -------
        A compiled Keras model with Adam Optimizer and Huber loss.
        """
        input = Input(shape=self._STATE_SPACE)
        x = Sequential([
            Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), padding='valid', activation='relu',
                    kernel_initializer=VarianceScaling(scale=2.0)),

            Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu', padding='valid',
                    kernel_initializer=VarianceScaling(scale=2.0)),

            Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid',
                    kernel_initializer=VarianceScaling(scale=2.0)),

            Conv2D(filters=1024, kernel_size=(7,7), strides=(1,1), activation='relu', padding='valid',
                    kernel_initializer=VarianceScaling(scale=2.0)),
        ])(input)

        value_tensor, advantage_tensor = Lambda(lambda x: tf.split(x, 2, axis=3))(x)

        value_tensor = Flatten()(value_tensor)
        advantage_tensor = Flatten()(advantage_tensor)

        advantage = Dense(self._NUM_ACTIONS, kernel_initializer=VarianceScaling(scale=2.0))(advantage_tensor)
        value = Dense(1, kernel_initializer=VarianceScaling(scale=2.0))(value_tensor)


        mean_advantage = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
        normalized_advantage = Subtract()([advantage, mean_advantage])

        output = Add()([value, normalized_advantage])

        model = Model(inputs=input, outputs=output)
        optimizer = Adam(1e-5)
        loss = Huber(delta=1.0)
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def update_replay_memory(self, transition):
        """
        Adds the provided transition to the `_replay_memory`.
        
        Args
        ----
        `transition` (tuple): A tuple of (curr_state, action, reward, next_state, terminal). terminal represents
                    whether the next_state is terminal or not.
        """
        self._replay_memory.append(transition)
    
    def train(self):
        """
        Trains the model with using a random `_BATCH_SIZE` sample of transitions from the _replay_memory. Training
        only happens when the size of the `_replay_memory` is greater than `_MIN_REPLAY_BUFFER_SIZE`.

        Returns
        -------
        A float denoting the loss obtained on the sample.
        """
        if len(self._replay_memory) < self._MIN_REPLAY_BUFFER_SIZE:
            return 0
        
        minibatch = random.sample(self._replay_memory, self._BATCH_SIZE)

        curr_states = np.array([transition[0] for transition in minibatch])/255.0
        curr_q_values = self._prediction_model.predict(curr_states)

        next_states = np.array([transition[3] for transition in minibatch])/255.0
        next_q_values_target = self._target_model.predict(next_states)
        # predict next state values using prediction model for Double-DQN implementation.
        next_q_values_predicted = self._prediction_model.predict(next_states)


        X, y = [], []

        for index, (curr_state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                new_q_value = reward
            else:
                new_q_value = reward + self._GAMMA*next_q_values_target[index, np.argmax(next_q_values_predicted[index])]
            
            curr_q_value = curr_q_values[index]
            curr_q_value[action] = new_q_value

            X.append(curr_state)
            y.append(curr_q_value)
        
        loss = self._prediction_model.train_on_batch(np.array(X)/255.0, np.array(y))
        return loss

    def update_target_weight(self):
        """
        Updates the target model weights by setting it to prediction model weights.
        """
        self._target_model.set_weights(self._prediction_model.get_weights())
    
    def save_checkpoint(self, episode, step, loss):
        """
        Saves the current training checkpoint of the model using the provided values. These checkpoints
        can be used to resume training. Checkpoints are saved only if the `_MODEL_NAME` was provided
        during initialization.

        Args
        ----
        `episode` (int): Episode till which the agent has been trained. (1 indexed)

        `step` (int): Step till which the agent has been trained. Each step denotes on transition seen or 1 action
            taken by the agent. (1 indexed)
        
        `loss`: Average loss observed in the current episode.
        """
        if self._MODEL_NAME is None:
            return
        
        self._start_episode = episode
        self._start_step = step
        with open(self._CHECKPOINT_FILE, 'w') as f:
            f.write(f'{self._start_episode},{self._start_step},{self._best_score},{self._epsilon}\n')
        
        q_values_hist = []
        for action in range(self._NUM_ACTIONS):
            avg = sum(self._q_values_history[action])/len(self._q_values_history[action])
            q_values_hist.append(round(avg, 3))
            q_values_hist.append(round(min(self._q_values_history[action]), 3))
            q_values_hist.append(round(max(self._q_values_history[action]), 3))
        
        # reset q_values buffer for next episode.
        self._q_values_history = [[] for action in range(self._NUM_ACTIONS)]
        
        q_values_hist = ','.join(str(q_values) for q_values in q_values_hist)
        with open(self._HISTORY_FILE, 'a') as f:
            f.write(f'{episode},{step},{self._train_score},{self._eval_score},{self._best_score}, \
            {self._epsilon:.5f},{loss},{q_values_hist}\n')

        self._save_models_and_buffer(episode)
        
        self.plot_history()
            

    def plot_history(self):
        """
        Plots the training, evaluation and best score in the `_HISTORY_FILE`. `_HISTORY_BUFFER_SIZE` is
        used to smooth the values by averaging over it.
        """
        train_history = deque(maxlen=self._HISTORY_BUFFER_SIZE)
        evaluate_history = deque(maxlen=self._HISTORY_BUFFER_SIZE)
        q_avg_history = [deque(maxlen=self._HISTORY_BUFFER_SIZE) for action in range(self._NUM_ACTIONS)]
        q_min_history = [deque(maxlen=self._HISTORY_BUFFER_SIZE) for action in range(self._NUM_ACTIONS)]
        q_max_history = [deque(maxlen=self._HISTORY_BUFFER_SIZE) for action in range(self._NUM_ACTIONS)]

        train = []
        evaluate = []
        best_evaluate = []
        loss = []
        episode = []
        q_value_avg = [[] for action in range(self._NUM_ACTIONS)]
        q_value_min = [[] for action in range(self._NUM_ACTIONS)]
        q_value_max = [[] for action in range(self._NUM_ACTIONS)]

        with open(self._HISTORY_FILE, 'r') as f:
            i = 0
            for line in f:
                i += 1 
                if i == 1: # ignore headers
                    continue
                ep, st, ts, es, bs, e, l = line.split(',')[:7]
                qv = line.split(',')[7:]
                
                train_history.append(float(ts))
                evaluate_history.append(float(es))
                for action in range(0, 3*self._NUM_ACTIONS, 3):
                    q_avg_history[action//3].append(float(qv[action]))
                    q_min_history[action//3].append(float(qv[action+1]))
                    q_max_history[action//3].append(float(qv[action+2]))

                episode.append(int(ep))
                train.append(sum(train_history)/len(train_history))
                evaluate.append(sum(evaluate_history)/len(evaluate_history))
                best_evaluate.append(float(bs))
                loss.append(float(l[:-1]))

                for action in range(0, 3*self._NUM_ACTIONS, 3):
                    q_value_avg[action//3].append(sum(q_avg_history[action//3])/len(q_avg_history[action//3]))
                    q_value_min[action//3].append(sum(q_min_history[action//3])/len(q_avg_history[action//3]))
                    q_value_max[action//3].append(sum(q_max_history[action//3])/len(q_avg_history[action//3]))


        plt.figure()
        plt.plot(episode, train, label='train_score')
        plt.plot(episode, evaluate, '--', label='eval_score')
        plt.plot(episode, best_evaluate, label='best_score')
        plt.legend(loc=2)

        plt.ylabel('score')
        plt.xlabel('episodes')
        plt.savefig(self._PROGRESS_FILE, dpi=self._PLOT_DPI)
        plt.close()

        plt.figure()
        plt.plot(episode, loss, label='loss')
        plt.legend(loc=2)
        plt.savefig(self._LOSS_FILE, dpi=self._PLOT_DPI)
        plt.close()

        plt.figure()
        for action in range(self._NUM_ACTIONS):
            plt.plot(episode, q_value_avg[action], label=f'q_avg_{action}')
            plt.plot(episode, q_value_min[action], label=f'q_min_{action}')
            plt.plot(episode, q_value_max[action], label=f'q_max_{action}')
        plt.legend(loc=2)
        plt.savefig(self._Q_VALUES_FILE, dpi=self._PLOT_DPI)
        plt.close()
    
    def save_best_model(self, score):
        """
        Saves the weights of the prediction model if the provided score is greater than the max of
        `_SAVE_SCORE_THRESHOLD` and the `_best_score`. Also updates the `_best_score` value.

        Args
        ----
        `score` (int): Current score of the agent.
        """
        if score > max(self._SAVE_SCORE_THRESHOLD, self._best_score):
            self._prediction_model.save_weights(
                os.path.join(self._MODEL_DIR, f'best_model_{score}.h5'))
        self._best_score = max(self._best_score, score)
    
    def _save_models_and_buffer(self, episode):
        """
        Saves the prediction and target models along with the replay buffer to disk.

        Args
        ----
        `episode` (int): Current episode, used to determine whether or not to save the replay buffer.
                         This is needed as buffer size grows to 2 GB at max capacity (100k) and takes
                         non-trivial time to write, so it is updated after every 'n' episode. Also, 
                         saving the buffer after each episode doesn't make much sense as most of the
                         data remains the same.
        """
        # save model to ensure both the model and its optimizer states are saved for better 
        # continuation of training
        self._prediction_model.save(self._PREDICTION_MODEL_WEIGHTS)
        
        # save only weights for target model as this model is not trained and only used for making
        #  predictions.
        self._target_model.save_weights(self._TARGET_MODEL_WEIGHTS)

        if episode % 100 == 0:
            with open(self._REPLAY_BUFFER_FILE, 'wb') as f:
                pickle.dump(self._replay_memory, f)
