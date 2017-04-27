# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

from collections import defaultdict as dd

Minimum of observed values is -9019.000000, at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                discount_fac  float      0.999990    
                dist_low      int        -111        
                epsilon       float      0.030200    
                mbot_high     int        40          
                mtbot_high    int        648         
                dist_high     int        1070        
                mtbot_low     int        -924        
                learning_rat  float      0.002363    
                mbot_low      int        468  

EPSILON = 0.0302
LEARNING_RATE = 0.002363 
DISCOUNT_FACTOR = 0.999990
MTTOP_LOW = -111 #DIST_LOW
MTTOP_HIGH = 1070
MTBOT_LOW = -924
MTBOT_HIGH = 648
MBOT_LOW = 468
MBOT_HIGH = 40
# nested 6 layers for tree_dist, tree_top, tree_bot, monkey_vel, monkey_top, monkey_bot

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.reset()
        self.count = 1
        # self.W = dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: [0,0]))))))
        #grav[1,4]
        #state space: velocity [+ or 0, 0 to -9 inclusive, lower]
        # dist: 0-25, 25.1-50, 50.1-100, 100-200, 200+
        # monkey_bot: 0-20, 20-60, 60-280, 280+
        # monkey_bot_to_tree_bot: -100, -40, -15, 0, 0-15, 15-40, 40-100, 100+ 
        #gravity (2) * velocity (3) * monkey_bot (4) * dist(5) *  monkey_bot_to_tree_bot (8) = ~? potential states

        # self.W = dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: [0,0])))))
        self.W = dd(lambda: dd(lambda: dd(lambda: [0,0]))) # minus gravity, velocity


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

    def get_W_parameters_array(self, state):
        '''
        Helper function for getting parameters needed to index into W table
        returns an array of size Depth of W
        '''
        cur_t = state['tree']
        cur_m = state['monkey']
        if not self.gravity:
            raise Exception("Need self.gravity defined before get_W_parameters_array() used")

# { 'score': <current score>,
#   'tree': { 'dist': <pixels to next tree trunk>,
#          'top': <height of top of tree trunk gap>,
#          'bot': <height of bottom of tree trunk gap> },
#   'monkey': { 'vel': <current monkey y-axis speed>,
#           'top': <height of top of monkey>,
#           'bot': <height of bottom of monkey> }}
        # grav = self.gravity == 4 #yes magic numbers

        # vel = 0
        # for limit in [-9, -999]:
        #     if cur_m['vel'] > limit:
        #         break
        #     vel += 1

        mbot = 0
        for limit in [MBOT_LOW, MBOT_HIGH, 999]:
            if cur_m['bot'] < limit:
                break
            mbot += 1

        # dist = 0
        # for limit in [238, 10, 999]:
        #     if cur_t['dist'] < limit:
        #         break
        #     dist += 1
            
        mttop = 0
        for limit in [MTTOP_LOW, MTTOP_HIGH, 999]:
            if cur_m['top'] - cur_t['top'] < limit:
                break
            mttop += 1

        mtbot = 0
        for limit in [MTBOT_LOW, MTBOT_HIGH, 999]:
            if cur_m['bot'] - cur_t['bot'] < limit:
                break
            mtbot += 1

        return [mbot, mttop, mtbot]


    def get_W_action_array(self, state):
        '''
        returns array of size 2 representing W values for each action for a particular state
        '''
        cur_t = state['tree']
        cur_m = state['monkey']

        W_params = self.get_W_parameters_array(state)
        cur_W = self.W
        for p in W_params:
            cur_W = cur_W[p]
        #cur_W should be array now
        assert len(cur_W) == 2
        return cur_W

    def get_policy(self, state):
        '''
        returns 0 or 1
        returns anticipated action based on state and self.W values by taking the maximum of the self.W values
        '''
        cur_t = state['tree']
        cur_m = state['monkey']

        W_actions = self.get_W_action_array(state)
        return W_actions.index(max(W_actions))

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

# { 'score': <current score>,
#   'tree': { 'dist': <pixels to next tree trunk>,
#          'top': <height of top of tree trunk gap>,
#          'bot': <height of bottom of tree trunk gap> },
#   'monkey': { 'vel': <current monkey y-axis speed>,
#           'top': <height of top of monkey>,
#           'bot': <height of bottom of monkey> }}

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump

        # calculate gravity if not yet done and can be done
        if not self.gravity and self.last_state:
            self.gravity = self.last_state['monkey']['vel'] - state['monkey']['vel']

        if not self.gravity:
            # first state; do nothing
            self.last_action = 0
            self.last_state  = state
            return self.last_action

        # SARSA off-policy model-free update
        if state and self.last_state and self.last_action is not None and self.last_reward is not None:
            last_t = self.last_state['tree']
            last_m = self.last_state['monkey']
            cur_t = state['tree']
            cur_m = state['monkey']
            
            W_last_actions = self.get_W_action_array(self.last_state)
            W_last_actions[self.last_action] -= LEARNING_RATE * (
                W_last_actions[self.last_action] - (
                    self.last_reward + DISCOUNT_FACTOR * self.get_W_action_array(state)[self.get_policy(state)]
                    )
                )
            a = self.get_W_parameters_array(self.last_state)
            # print '%i :updated %i %i %i to %f' % (self.count, a[0], a[1], a[2], W_last_actions[self.last_action])
            self.count += 1
            if self.last_reward < 0:
                print 'DEAD with %i' % self.last_reward 
                print 'Score was %i' % state['score']
            # W_actions = self.W[last_t['dist']][t['top']][t['bot']][m['vel']][m['top']][m['bot']]
            # self.W[last_t['dist']][last_t['top']][last_t['bot']][last_m['vel']][last_m['top']][last_m['bot']][self.last_action] -= LEARNING_RATE * (
            #     self.W[last_t['dist']][last_t['top']][last_t['bot']][last_m['vel']][last_m['top']][last_m['bot']][self.last_action] - (
            #         self.last_reward + DISCOUNT_FACTOR * self.W[cur_t['dist']][cur_t['top']][cur_t['bot']][cur_m['vel']][cur_m['top']][cur_m['bot']][self.get_policy(state)]
            #         )
            #     )

        # epsilon-greedy policy for minimizing loss, balancing exploration vs. exploitation
        # but no exploration past 10 b/c only mistakes then
        if state['score'] < 10 and npr.rand() < EPSILON:
            new_action = npr.rand() < 0.5
        else:
           new_action = self.get_policy(state)

        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    score = 0
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        score += swing.score

        # Reset the state of the learner.
        learner.reset()
    print score
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 20, 1)

    # Save history. 
    np.save('hist',np.array(hist))


