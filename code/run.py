# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

from collections import defaultdict as dd

EPSILON = 0.1
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
# nested 6 layers for tree_dist, tree_top, tree_bot, monkey_vel, monkey_top, monkey_bot

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.W = dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: [0,0]))))))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.W = dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: dd(lambda: [0,0]))))))

	def get_policy(self, state):
		'''
		returns 0 or 1
		returns anticipated action based on state and self.W values by taking the maximum of the self.W values
		'''
		W_actions = self.W[t['dist']][t['top']][t['bot']][m['vel']][m['top']][m['bot']]
		return W_actions.index(max(W_actions))

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

# { ’score’: <current score>,
#   ’tree’: { ’dist’: <pixels to next tree trunk>,
#          ’top’: <height of top of tree trunk gap>,
#          ’bot’: <height of bottom of tree trunk gap> },
#   ’monkey’: { ’vel’: <current monkey y-axis speed>,
#           ’top’: <height of top of monkey>,
#           ’bot’: <height of bottom of monkey> }}

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump


    	# SARSA off-policy model-free update
        if state and self.last_state and self.last_action and self.last_reward:
        	last_t = self.last_state['tree']
        	last_m = self.last_state['monkey']
        	cur_t = self.state['tree']
        	cur_m = self.state['monkey']

        	# W_actions = self.W[last_t['dist']][t['top']][t['bot']][m['vel']][m['top']][m['bot']]
        	self.W[last_t['dist']][last_t['top']][last_t['bot']][last_m['vel']][last_m['top']][last_m['bot']][self.last_action] -= LEARNING_RATE * (
        		self.W[last_t['dist']][last_t['top']][last_t['bot']][last_m['vel']][last_m['top']][last_m['bot']][self.last_action] - (
        			self.last_reward + DISCOUNT_FACTOR * self.W[cur_t['dist']][cur_t['top']][cur_t['bot']][cur_m['vel']][cur_m['top']][cur_m['bot']][self.get_policy(state)]
        			)
        		)

        # epsilon-greedy policy for minimizing loss, balancing exploration vs. exploitation
        if npr.rand() < EPSILON:
	        new_action = npr.rand() < EPSILON
	    else:
	    	new_action = get_policy(state)

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

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


