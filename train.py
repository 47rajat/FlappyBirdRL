import numpy as np
from agent import FlappyBirdRLAgent
from environment import FlappyBirdEnv
import utils
from collections import deque
import time
import os
import numpy as np

# Number of games the agent plays while training.
NUM_EPISODES = 50_000

# Used to smooth the average training score by averaging.
SCORE_HISTORY_LEN = 100

# Step interval in which agent's target model is updated.
UPDATE_EVERY = 10_000

# Episode interval in which agent's prediction model is evaluated.
EVALUATE_EVERY = 5

env = FlappyBirdEnv(display=True)
agent = FlappyBirdRLAgent(model_name='duel_ddqn', epsilon=1.0, actions=env.NUM_ACTIONS, state_space=env.STATE_SPACE)


train_score_history = deque(maxlen=SCORE_HISTORY_LEN)
start_episode, step = agent.get_start_checkpoint()
assert(start_episode >= 0 and step >= 0)

# Explore till the agent buffer size is greater than _MIN_REPLAY_BUFFER_SIZE.
buffer_size = len(agent._replay_memory)
while buffer_size < agent._MIN_REPLAY_BUFFER_SIZE:
    done = False
    curr_state = env.reset()
    while not done:
        buffer_size += 1
        # agent game-play logic
        action = np.random.randint(env.NUM_ACTIONS)
        next_state, reward, done, score  = env.step(action)
        agent.update_replay_memory((curr_state, action, reward, next_state, done))
        curr_state = next_state
    print(f'[explore] buffer_size: {buffer_size}, score: {score}')


# Train agent for a max of NUM_EPISODES.
for episode in range(start_episode, NUM_EPISODES):
    num_actions = 0
    loss = 0
    curr_state = env.reset()
    done = False
    while not done:
        step += 1
        if step % (2*agent._EPSILON_DECAY_STEPS) == 0:
            agent._epsilon = 1e-1 # for cyclic exploration.
        
        if step % UPDATE_EVERY == 0:
            agent.update_target_weight()
        
        # agent game-play logic
        action = agent.get_action(curr_state)
        next_state, reward, done, score = env.step(action)
        agent.set_train_score(score)
        agent.update_replay_memory((curr_state, action, reward, next_state, done))
        num_actions += 1
        loss += agent.train()
        curr_state = next_state
    
    agent.set_train_score(score)
    train_score_history.append(score)
    avg_train_score = sum(train_score_history)/len(train_score_history)
    print(f'[train] ep: {episode+1}/{NUM_EPISODES}, steps: {step}, avg_score: {avg_train_score:.2f}, eps: {agent._epsilon:.5f}')

    loss /= num_actions
    agent.save_checkpoint(episode + 1, step, loss)


    # evaluate agent.
    if (episode+1) % EVALUATE_EVERY == 0:
        curr_state = env.reset()
        done = False
        while not done:
            # agent game-play logic
            action = agent.get_greedy_action(curr_state)
            next_state, reward, done, score = env.step(action)
            agent.set_eval_score(score)
            agent.update_replay_memory((curr_state, action, reward, next_state, done))
            curr_state = next_state
        
        agent.set_eval_score(score)
        print(f'[eval] episode: {episode+1}, score: {score}')
 
        agent.save_best_model(agent.get_eval_score())