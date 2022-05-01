import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle #save and load  q table
from matplotlib import style
import time

style.use("ggplot")

# SIZE = 10#grid size 10x10
#scale 
s=2
SIZE = 20*s #grid size 10x10

HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
# SHOW_EVERY = 3000  # how often to play through env visually.
SHOW_EVERY = 1  # how often to play through env visually.

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
#bgr scheme
d = {1: (255, 175, 0),
     2: (0, 255, 0),#green
     3: (0, 0, 255)}#enemy, red


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):#distances between elements
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

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
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):#distances between elements
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

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
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
#     # initialize the q-table#
    q_table = {}
    # q_table = np.zeros((2*SIZE,2*SIZE,2*SIZE, 2*SIZE),dtype=float)

    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                        #deltas: player to food, player to enemy
                        # q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]
                        q_table[((i, ii), (iii, iiii))] = [0 for i in range(4)]

# else:
#     with open(start_q_table, "rb") as f:
#         q_table = pickle.load(f)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    

    # sofa1 = Blob()
    # enemy.x = 10
    # enemy.y = 10

    LAST=SIZE-1

    #y rightwards, x downwards
    sofa = []
    for i in range(LAST-2*s, LAST-1*s+1):
        for j in range(2*s, 5*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            sofa.append(cell)

    sofa1 = []
    for i in range(LAST-6*s, LAST-3*s+1):
        for j in range(6*s, 7*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            sofa1.append(cell)

    wall1 = []
    for i in range(LAST-7*s, LAST-1*s+1):
        for j in range(8*s, 8*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            wall1.append(cell)

    wall2 = []
    for i in range(LAST-10*s, LAST-10*s+1):
        for j in range(8*s, 19*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            wall2.append(cell)

    wall3 = []
    for i in range(1*s, LAST-10*s+1):
        for j in range(5*s, 5*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            wall3.append(cell)
    
    slab_v = []
    for i in range(1*s, LAST-10*s+1):
        for j in range(1*s, 1*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            slab_v.append(cell)

    slab_h = []
    for i in range(1*s, 2*s+1):
        for j in range(1*s, 4*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            slab_h.append(cell)

    table = []
    for i in range(LAST-6*s, LAST-4*s+1):
        for j in range(3*s, 4*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            table.append(cell)

    bed_br = []
    for i in range(LAST-5*s, LAST-2*s+1):
        for j in range(12*s, 19*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            bed_br.append(cell)

    bed_top = []
    for i in range(1*s, 6*s+1):
        for j in range(10*s, 12*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            bed_top.append(cell)

    wall4 = []
    for i in range(1*s, 6*s+1):
        for j in range(16*s, 16*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            wall4.append(cell)

    almirah = []
    for i in range(LAST-9*s, LAST-8*s+1):
        for j in range(13*s, 16*s+1):
            cell = Blob()
            cell.x = i
            cell.y = j
            almirah.append(cell)
    
    boundary = []
    for i in range(0, SIZE):
        #vertical-left boundary
        brick_vl = Blob()
        brick_vl.x = i*s
        brick_vl.y = 0
        boundary.append(brick_vl)

        brick_vr = Blob()
        brick_vr.x = i*s
        brick_vr.y = SIZE-1*s
        boundary.append(brick_vr)

        #horizontal-top boundary
        brick_ht = Blob()
        brick_ht.x = SIZE-1*s
        brick_ht.y = i*s
        boundary.append(brick_ht)

        brick_hb = Blob()
        brick_hb.x = 0
        brick_hb.y = i*s
        boundary.append(brick_hb)

    enemy = Blob()

    # enemy.x = 10
    # enemy.y = 10

    enemy2 = Blob()
    enemy2.x = 9*s
    enemy2.y = 2*s

    enemy3 = Blob()
    enemy3.x = 1*s
    enemy3.y = 8*s

    enemy4 = Blob()
    enemy4.x = 1*s
    enemy4.y = 14*s

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    #start steps
    episode_reward = 0
    for i in range(200*s):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            for i in range(0, len(boundary)):
                env[boundary[i].x][boundary[i].y] = d[ENEMY_N]
            for i in range(0, len(sofa)):
                env[sofa[i].x][sofa[i].y] = d[ENEMY_N]
            for i in range(0, len(sofa1)):
                env[sofa1[i].x][sofa1[i].y] = d[ENEMY_N]
            for i in range(0, len(wall1)):
                env[wall1[i].x][wall1[i].y] = d[ENEMY_N]
            for i in range(0, len(wall2)):
                env[wall2[i].x][wall2[i].y] = d[ENEMY_N]
            for i in range(0, len(wall3)):
                env[wall3[i].x][wall3[i].y] = d[ENEMY_N]
            for i in range(0, len(wall4)):
                env[wall4[i].x][wall4[i].y] = d[ENEMY_N]
            for i in range(0, len(table)):
                env[table[i].x][table[i].y] = d[ENEMY_N]
            for i in range(0, len(bed_br)):
                env[bed_br[i].x][bed_br[i].y] = d[ENEMY_N]
            for i in range(0, len(bed_top)):
                env[bed_top[i].x][bed_top[i].y] = d[ENEMY_N]
            for i in range(0, len(almirah)):
                env[almirah[i].x][almirah[i].y] = d[ENEMY_N]
            for i in range(0, len(slab_v)):
                env[slab_v[i].x][slab_v[i].y] = d[ENEMY_N]
            for i in range(0, len(slab_h)):
                env[slab_h[i].x][slab_h[i].y] = d[ENEMY_N]

            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            env[enemy2.x][enemy2.y] = d[ENEMY_N]  # sets the enemy location to red
            env[enemy3.x][enemy3.y] = d[ENEMY_N]  # sets the enemy location to red
            env[enemy4.x][enemy4.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)