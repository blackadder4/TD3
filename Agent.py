import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from UTILS.Replay_Buffer import ReplayBuffer
from TD3_network import ActorNetwork, CriticNetwork
from UTILS.Noise import OUActionNoise

class Agent:
    def __init__(self, alpha=0.0001, beta=0.001, gamma=0.99, n_actions=4,
                 fc1_dims=400, fc2_dims=300, max_size=1000000, tau=0.005,
                 batch_size=100, noise_sigma=0.2, checkpoint='Models/', env=None,
                 noise_val = 0.03,noise_mu=0.0, noise_theta=0.15, noise_dt=1e-2, OU = True, 
                 update_interval = 2, warmup = 1000):

        # Init actor, critic, and the subsequent target network
        self.actor = ActorNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        #now we have two critics
        self.critic_1 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_2 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_actor = ActorNetwork(n_actions=n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_critic_1 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_critic_2 = CriticNetwork(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.noise_val = noise_val
        self.time_step = 0
        self.learn_step_cntr = 0
        self.gamma = gamma
        self.n_actions = n_actions
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.update_interval = update_interval
        self.warmup = warmup
        self.tau = tau
        self.checkpoint = checkpoint
        #targets are not compile because they will be only forward prop to get the gamma Q(s,a)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

        # Initialize OUActionNoise with the provided parameters
        self.replay_buffer = ReplayBuffer(max_size, env.observation_space.shape, env.action_space.shape[0], batch_size)
        
        # Rest of your code...

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic_1.set_weights(weights)
                
        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic_2.set_weights(weights)


    def choose_action(self, observation, eval_mode=False):
        #states are noised before adding into the actions space
        #modifications 1 , OU noise are not temp replaced with gaussian nosie
        #modifications 2 , Eval mode is temp removed
        #This is the step to add noise to the states space and then clip my the legal values then pass into the policy mu
        if self.time_step < self.warmup:
            mu = np.random.normal(scale = self.noise_val , size= (self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation])
            mu = self.actor(state)[0]
        if not eval_mode:
            #remember we are not adding noise to the states instead of that exploration noise approach
            # if it is now in eval mode we add the noise to the action space then clip
            mu_prime = mu + np.random.normal(scale = self.noise_val)
            mu_prime = tf.clip_by_value(mu_prime,self.min_action,self.max_action)
        self.time_step += 1

        
        return mu_prime


    def learn(self,debug = False):
        if len(self.replay_buffer) < self.batch_size:
            #this should make sure the buffer is big enough
            return
        #from testing returning done as pure tensor is fine
        #states,actions,rewards,new_states,done = self.replay_buffer.sample(tensor = True, sample_size = self.batch_size)
        dictionary = self.replay_buffer.sample(tensor = False, sample_size = self.batch_size)
        states = dictionary['states']
        action = dictionary['actions']
        rewards = dictionary['rewards']
        new_state = dictionary['next_states']
        done = dictionary['dones']
        with tf.GradientTape(persistent = True) as tape:
            #plug in states into actor network
            new_policy_actions = self.target_actor(states)
            noise = np.random.normal(scale=self.noise_val, size=new_policy_actions.shape)
            new_policy_actions += noise
            new_policy_actions = tf.clip_by_value(new_policy_actions, self.min_action, self.max_action)

            #print("States Shape:", states.shape)
            #print("New Policy Actions Shape:", new_policy_actions.shape)
            #squeeze kill dimensions
            Q_target_val_1 = tf.squeeze(self.target_critic_1(states,new_policy_actions),1)
            Q_target_val_2 = tf.squeeze(self.target_critic_2(states,new_policy_actions),1)

            Q_val_1 = tf.squeeze(self.critic_1(states,new_policy_actions), 1)
            Q_val_2 = tf.squeeze(self.critic_2(states,new_policy_actions), 1)

            critic_val = tf.math.minimum(Q_target_val_1,Q_target_val_2)
            target = rewards + self.gamma * critic_val * (1-done)
            #the critic val is the one that needed the min of the two critics
            #the MSE is still updating both critic
            critic_loss_1 = keras.losses.MSE(target,Q_val_1)
            critic_loss_2 = keras.losses.MSE(target,Q_val_2)

        #Apply gradient descent to the params 1 and 2
        params_1 = self.critic_1.trainable_variables
        params_2 = self.critic_2.trainable_variables
        grads = tape.gradient(critic_loss_1, params_1)
        self.critic_1.optimizer.apply_gradients(zip(grads,params_1))
        grads = tape.gradient(critic_loss_2, params_2)
        self.critic_2.optimizer.apply_gradients(zip(grads,params_2))
        ++self.learn_step_cntr
        #remember now we update the actor lagging behind the critics
        if self.learn_step_cntr % self.update_interval == 0:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states)
                critic_1_val = self.critic_1(states,new_actions)
                #reduce loss in terms of the optimization of the gradient of the critic val
                actor_loss = -tf.math.reduce_mean(critic_1_val)
                actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
            #soft update the target networks
            self.update_network_parameters()


    def store_transition(self, state, action, reward, new_state, done,debug = False):
        if debug:
            print("transition storing")
            print(state, action, reward, new_state, done)
        action = tf.squeeze(action).numpy()
        self.replay_buffer.add(state, action, float(reward), new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.checkpoint + 'actor')
        self.critic_1.save(self.checkpoint + 'critic_1')
        self.critic_2.save(self.checkpoint + 'critic_2')
        self.target_actor.save(self.checkpoint + 'target_actor')
        self.target_critic_1.save(self.checkpoint + 'target_critic_1')
        self.target_critic_1.save(self.checkpoint + 'target_critic_2')
        print('... models saved ...')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.checkpoint + 'actor')
        self.critic_1 = keras.models.load_model(self.checkpoint + 'critic_1')
        self.critic_2 = keras.models.load_model(self.checkpoint + 'critic_2')
        self.target_actor = keras.models.load_model(self.checkpoint + 'target_actor')
        self.target_critic_1 = keras.models.load_model(self.checkpoint + 'target_critic_1')
        self.target_critic_2 = keras.models.load_model(self.checkpoint + 'target_critic_2')
        print('... models loaded ...')
