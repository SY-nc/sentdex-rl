import numpy as np
#import keras.backend.tensorflow_backend as backend
import keras.backend as backend

from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
# can use _ in place of , # we take a random sampling of those 50,000 steps and thats the batch to train the NN (provide stability & smoothing)
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256' # whatever you want, as here NN 
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20 # not using right now

# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
# AGGREGATE_STATS_EVERY = 50  #  episodes (step aggregation)
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = False # set to true if we want to actually see the visuals of everything running 
# SHOW_PREVIEW = True

# ADD BLOB CLASS (Copy & Paste) 
class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

### ADD Blob Environment (copy & paste)

class BlobEnv:
    # SIZE = 50
    s = 1
    SIZE = 20*s
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    # ENEMY_PENALTY = np.Inf
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        s = self.s
        SIZE = self.SIZE
        self.enemy = []
        #define obstacles here

        LAST=self.SIZE-1
        # sofa = []
        for i in range(LAST-2*s, LAST-1*s+1):
            for j in range(2*s, 5*s+1):
                #definition of Blob requires an argument for SIZE
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # sofa1 = []
        for i in range(LAST-6*s, LAST-3*s+1):
            for j in range(6*s, 7*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # wall1 = []
        for i in range(LAST-7*s, LAST-1*s+1):
            for j in range(8*s, 8*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # wall2 = []
        for i in range(LAST-10*s, LAST-10*s+1):
            for j in range(8*s, 19*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # wall3 = []
        for i in range(1*s, LAST-10*s+1):
            for j in range(5*s, 5*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)
        
        # slab_v = []
        for i in range(1*s, LAST-10*s+1):
            for j in range(1*s, 1*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # slab_h = []
        for i in range(1*s, 2*s+1):
            for j in range(1*s, 4*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # table = []
        for i in range(LAST-6*s, LAST-4*s+1):
            for j in range(3*s, 4*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # bed_br = []
        for i in range(LAST-5*s, LAST-2*s+1):
            for j in range(12*s, 19*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # bed_top = []
        for i in range(1*s, 6*s+1):
            for j in range(10*s, 12*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # wall4 = []
        for i in range(1*s, 6*s+1):
            for j in range(16*s, 16*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)

        # almirah = []
        for i in range(LAST-9*s, LAST-8*s+1):
            for j in range(13*s, 16*s+1):
                cell = Blob(SIZE)
                cell.x = i
                cell.y = j
                self.enemy.append(cell)
        
        # boundary = []
        for i in range(0, SIZE):
            #vertical-left boundary
            brick_vl = Blob(SIZE)
            brick_vl.x = i*s
            brick_vl.y = 0
            self.enemy.append(brick_vl)

            brick_vr = Blob(SIZE)
            brick_vr.x = i*s
            brick_vr.y = SIZE-1*s
            self.enemy.append(brick_vr)

            #horizontal-top boundary
            brick_ht = Blob(SIZE)
            brick_ht.x = SIZE-1*s
            brick_ht.y = i*s
            self.enemy.append(brick_ht)

            brick_hb = Blob(SIZE)
            brick_hb.x = 0
            brick_hb.y = i*s
            self.enemy.append(brick_hb)

        enemy2 = Blob(SIZE)
        enemy2.x = 9*s
        enemy2.y = 2*s
        self.enemy.append(enemy2)

        enemy3 = Blob(SIZE)
        enemy3.x = 1*s
        enemy3.y = 8*s
        self.enemy.append(enemy3)

        enemy4 = Blob(SIZE)
        enemy4.x = 1*s
        enemy4.y = 14*s
        self.enemy.append(enemy4)
        
        self.player = Blob(SIZE)
        for i in range (0, len(self.enemy)):
            while self.player == self.enemy[i]:
                self.player = Blob(self.SIZE)
        
        self.food = Blob(SIZE)
        #to prevent initializing food on player
        for i in range (0, len(self.enemy)):
            while self.food == self.player or self.food == self.enemy[i]:
                self.food = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #self.enemy.move()
        #self.food.move()
        ##############


        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        for i in range (0, len(self.enemy)):
            if self.player == self.enemy[i]:
                reward = -self.ENEMY_PENALTY
            elif self.player == self.food:
                reward = self.FOOD_REWARD
            else:
                reward = -self.MOVE_PENALTY

        done = False
        #if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 1250:
            done = True

        return new_observation, reward, done

    def render(self): # RENDERING
        img = self.get_image()
        #img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        #img = img.resize((700, 700))  # resizing so we can see our agent in all its glory.
        img = img.resize((300, 300), resample=Image.BOX) # to remove blurry boxes
        # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpen = cv2.filter2D(img, -1, sharpen_kernel)
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN # Pull an exact image from our environment (image as input, we are not doing the DELTA to food or enemy)
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        for i in range(0,len(self.enemy)):
            env[self.enemy[i].x][self.enemy[i].y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = BlobEnv()

# For stats
ep_rewards = [-200] # some initial things

# For more repetitive results
random.seed(1)
np.random.seed(1)
# tf.set_random_seed(1)
tf.random.set_seed(1)

## TO TRAIN THE MULTIPLE MODEL ON SAME MACHINE USE THIS CODE 
# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# ADDED Tensorboard class (from net sol.)## AttributeError: 'ModifiedTensorBoard' object has no attribute '_write_logs'
# (modifying the tensorboard functionalities from Tensorflow and Keras): copied from text 
class ModifiedTensorBoard(TensorBoard):     
    
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)                                     
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
    
    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
    
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False
    
    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    
    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass
    
    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()

# Own Tensorboard class ## AttributeError: 'ModifiedTensorBoard' object has no attribute '_write_logs'
# class ModifiedTensorBoard(TensorBoard):

#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary.create_file_writer(self.log_dir)

#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass

#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)

#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass

#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass


#     def _write_logs(self, logs, index):
#         with self.writer.as_default():
#             for name, value in logs.items():
#                 tf.summary.scalar(name, value, step=index)
#                 self.step += 1
#                 self.writer.flush()
    
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self, model_path): # init method for this model 


    ## Two models: for stability and consistency to learn something because initially so much randomness

        # Main model # gets trained every step
        ##[satish, 2022-05-11] if model_path is not blank, load the model to continue training on it, 
        #   otherwise, setup a new model
        if not len(model_path) == 0:
            print("loading model from : "+ str(model_path))
            self.model = models.load_model(model_path)
            print("model load successful!")
        else:
            self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model() # This is the model that we are doing .predict every step 
        self.target_model.set_weights(self.model.get_weights()) # same weights (After every some no. of episodes you will re-update your target model)
 
        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) # deque-think of it as an array or a list 

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time()))) # for keep tracking (3.5 or younger Python you can't do f strings)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0 # to track internally when we are ready to update that target model network with main network's weights

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu')) # 256 convolutions $$ convolution window of 3*3
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2)) # dropout 20%

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors (# flatten here so we can pass it through dense layers (this converts 3D feature maps to 1D feature vectors))
        model.add(Dense(64)) # 64 layers

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy']) # we will track accuracy
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    ### IN video it is there.
    # def get_qs(self, state):
    #     return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    ###
     
    ## Add to 5.1 
    # add new train method
    # Trains main network every step during episode
    def train(self, terminal_state, step):


        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255 # /255 to scale or normalize 
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255 # after we take steps
        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # feature sets (images from the game)
        y = [] # labels or our targets (action we decide to take)

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch): # calculate learned Q-value, done with environment 

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index]) 
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index] # update that Q-value 
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state) # inputs
            y.append(current_qs) # outputs

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # shuffle=False because we have already grabbed a random sampling, custom callback, we will fit all of this if on our terminal state O/W we will fit nothing

        # Update target network counter every episode (updating to determine if we want to update target_model yet)
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights()) # we just copy over the weights from our initial model
            self.target_update_counter = 0 # reset to zero 

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

## ADD: to do iteration and all
## [satish, 2022-05-11] to continue training an existing model, 
#       initialize the class 'DQNAgent' with argument model_path
agent = DQNAgent("models/6.1.3-400-ep-done.model")

#[satish, 2022-05-11] if training is to be started, set this to zero
episodes_done = 400

if episodes_done == 0:
    print("starting training from the beginning...")
else:
    print("continuing training from episode: " + str(episodes_done))

# Iterate over episodes
for episode in tqdm(range(episodes_done + 1, EPISODES + 1), ascii=True, unit='episodes'): # ascii for windows fellows

    print("current episode: " + str(episode))
    
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1 # say we are on currently step no 1

    # Reset environment and get initial state
    current_state = env.reset() # kind of matching the syntax from OpenAIGym

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done)) # every step we have to add that to the replay memory
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        # if min_reward >= MIN_REWARD:
        if episode % 50 == 0:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
