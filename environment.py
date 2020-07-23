from ple.games.flappybird import FlappyBird
from ple import PLE
import cv2
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import time
import utils

np.random.seed(123)

class FlappyBirdEnv:
    """ 
    This is the Reinforcement Learning Environment that wraps the PLE Flappy Bird Game. The RL agent 
    interacts with the environment by providing which action it wants to take in the current state. 
    The environment in turn provides the reward and the next state to agent after executing the provided
    action.
    """
    def __init__(self, display=False):
        """
        Initializes a new environment for FlappyBird game.
        """
        game = game = FlappyBird()
        self._game = PLE(game, fps=30, display_screen=display)
        
        # _display_game flag controls whether or not to render the state that is being provided by the 
        # environment.
        self._display_game = display
        
        if self._display_game:
            self._display = self.show_img() # display sets up a cv2 window where the current state is displayed.
            self._display.__next__() # iterate over the display generator.
        
        self.NUM_ACTIONS = len(self._game.getActionSet()) # defines the number of action agent can take in the environment.

        self._ACTION_MAP = {}
        for i, action in enumerate(self._game.getActionSet()):
            self._ACTION_MAP[i] = action
        
        # Number contiguous images the environment provides as state. Basically at any time, the
        # environment provides a stack of last 4 (including the current) images as the state to the agent.
        self._IMAGE_STACK_SIZE = 4

        # Dimension of the (greyscale) image provided as state.
        self._PROCESSED_IMAGE_SIZE = 84

        # Determines the number of times the provided action is executed before returning the next
        # state.
        self._SKIP_FRAMES = 4 

        # Used by the RL agent to set up it's CNN model.
        self.STATE_SPACE = (self._PROCESSED_IMAGE_SIZE, self._PROCESSED_IMAGE_SIZE, self._IMAGE_STACK_SIZE)
        self._init_states()

    def _init_states(self):
        """
        Initializes/Resets the states for the environment.
        """
        self._image_stack = None # holds the current state, i.e., stack of 4 images.
        self._score = 0
    
    def step(self, action):
        """
        Provides the next state and rewards after executing the provided action.

        Args
        ------
        `action` (int): Action to be taken from the current state.
        """
        reward = 0
        for i in range(self._SKIP_FRAMES):
            reward += self._game.act(self._ACTION_MAP[action])
        

        done = self._game.game_over()
        self._score += reward
        
        clipped_reward = self._clip_reward(reward)

        self.grab_screen()
        if self._display_game:
            self._display.send(self._image_stack) # display image on the screen
        
        return (self._image_stack.copy(), clipped_reward, done, self._score) 
    
    def _clip_reward(self, reward):
        """
        Clips the provided reward between [-1, 1]

        Args
        ----
        `reward` (float): The reward that is to be clipped.

        Returns
        -------
        A float represent the clipped reward.
        """
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        
        return reward
    
    def reset(self):
        """
        Resets the game and provides the starting state.

        Returns
        -------
        A numpy `_IMAGE_STACK_SIZE`-d numpy array (or greyscale image) representing the current state
        of the environment
        """
        self._game.reset_game()
        
        self._init_states()
        
        self.grab_screen()
        if self._display_game:
            self._display.send(self._image_stack)

        return self._image_stack.copy()

    def show_img(self):
        '''
        Show current state (`_IMAGE_STACK_SIZE` greyscale images) in an opencv window.

        Returns
        -------
        A generator that to which the images can be sent for displaying.
        '''
        return utils.show_image('Model Input (4 images)')
    
    def grab_screen(self):
        """
        Grabs 1 to _IMAGE_STACK_SIZE images (depending upon whether called after reseting or not) and
        adds it to the image_stack in chronological order, i.e., most recent image is the last.
        """
        if self._image_stack is None:
            self._image_stack = np.zeros(self.STATE_SPACE, dtype=np.uint8)
            for i in range(self._IMAGE_STACK_SIZE):
                self._game.act(None)
                self._image_stack[:, :, i] = self.get_processed_image()
        else:
            self._image_stack[:, :, :self._IMAGE_STACK_SIZE-1] = self._image_stack[:, :, 1:]
            self._image_stack[:, :, self._IMAGE_STACK_SIZE-1] = self.get_processed_image()
    
    def get_processed_image(self):
        """
        Fetches the current gameplay screenshot and processes it.

        Returns
        -------
        A processed greyscale image (as numpy array) representing the current gameplay state.
        """
        screen = self._game.getScreenGrayscale()
        image = self.process_image(screen)
        return image

    def process_image(self, image):
        """
        Processes the input image by performing following steps:
        i. Cropping and transposing the image to obtain the Region Of Interest (ROI)
        ii. Resizing the ROI to (`_PROCESSED_IMAGE_SIZE`, `_PROCESSED_IMAGE_SIZE`) dimension.

        Args
        ----
        `image` (numpy array): The image which is to be processed.

        Returns
        -------
        A processed greyscale image (as numpy array).
        """
        # Step 1.
        image = image[:, :410]
        image = np.transpose(image, (1, 0))
        
        # Step 2.
        image = cv2.resize(image, (self._PROCESSED_IMAGE_SIZE, self._PROCESSED_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        return image
