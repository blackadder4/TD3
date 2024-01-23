import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LayerNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
class CriticNetwork(keras.Model):
    def __init__(self,fc1_dims = 1024, fc2_dims = 512):
    	super(CriticNetwork,self).__init__()
    	self.fc1 = Dense(fc1_dims,activation = 'relu')
    	self.fc2 = Dense(fc2_dims,activation = 'relu')
    	self.q = Dense(1,activation = None)

    def call(self,state, action):
    	#breaks inputs into state and action
    	#remember that this is the Q network and we are here to take the Q(s,a)~V(s)
    	x = self.fc1(tf.concat([state,action],axis = 1))
    	x = self.fc2(x)
    	q = self.q(x)

    	return q

class ActorNetwork(keras.Model):
	def __init__(self, n_actions, fc1_dims = 1024, fc2_dims = 512):
		super(ActorNetwork,self).__init__()
		self.fc1 = Dense(fc1_dims, activation = 'relu')
		self.fc2 = Dense(fc2_dims, activation = 'relu')
		self.mu  = Dense(n_actions, activation = 'tanh')

	def call(self, state):
		x = self.fc1(state)
		x = self.fc2(x)
		mu = self.mu(x)

		return mu