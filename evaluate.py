import numpy as np
from agent import FlappyBirdRLAgent
from environment import FlappyBirdEnv
import utils
from collections import deque
import time
import os
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')

# Number of times evaluation is performed.
NUM_TRIALS = 3

# Number of games the agent plays for evaluation per trial.
NUM_EPISODES = 100

# Location where the result will be saved.
RESULT_PATH = './results/duel_ddqn_new'

# Path to model weights, which will be used for evaluation.
EVAL_MODEL_PATH = './models/duel_ddqn/target_model_weights.h5'

env = FlappyBirdEnv(display=True)
agent = FlappyBirdRLAgent(eval_model=EVAL_MODEL_PATH, actions=env.NUM_ACTIONS, state_space=env.STATE_SPACE)


for trial in range(NUM_TRIALS):
    eval_score_history = []
    for episode in range(NUM_EPISODES):
        curr_state = env.reset()
        done = False
        while not done:
            # agent game-play logic
            action = agent.get_greedy_action(curr_state)
            next_state, reward, done, score = env.step(action)
            agent.set_eval_score(score)
            curr_state = next_state
        
        eval_score_history.append(score)
        print(f'[eval] trial: {trial+1}, episode: {episode+1}, score: {score}')

    assert(False)
    # plot result.
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    avg_score = sum(eval_score_history)/len(eval_score_history)
    episodes = np.arange(1, NUM_EPISODES + 1)

    plt.figure()
    plt.plot(episodes, eval_score_history, label='eval_score')
    plt.plot(episodes, [avg_score for i in range(len(episodes))], '-', label=f'avg_eval_score={avg_score}')
    plt.xlabel('# episodes')
    plt.ylabel('score')
    plt.title(f'Trial {trial+1}')
    plt.legend(loc=2)
    plt.savefig(f'{RESULT_PATH}/result_{trial+1}.png', dpi=240)
    plt.close()