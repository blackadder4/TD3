import numpy as np
import tensorflow as tf

class ReplayBuffer:
    """
    A circular buffer for storing and sampling experiences for reinforcement learning.

    This buffer stores experiences which consist of state, action, reward, next_state, and done flag.
    It supports storing experiences of varying sizes, making it adaptable to different environments.

    Parameters:
    buffer_size : int
        The maximum size of the buffer.
    state_size : int or tuple
        The size of the state.
    action_size : int
        The size of the action.
    sample_size : int
        The number of samples to draw from the buffer.

    Attributes:
    buffer_size : int
        The maximum size of the buffer.
    state_size : int or tuple
        The size of the state.
    action_size : int
        The size of the action.
    sample_size : int
        The number of samples to draw from the buffer.
    buffer : dict
        The storage for states, actions, rewards, next_states, and dones.
    ptr : int
        The current position in the buffer to insert the next experience.
    size : int
        The current size of the buffer.
    """

    def __init__(self, buffer_size, state_size, action_size, sample_size):
    	#check for the correct states and action size values, no neg no zero allowed
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError("Buffer size must be a positive integer.")
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("Sample size must be a positive integer.")

        if not (isinstance(state_size, int) or (isinstance(state_size, tuple) and all(isinstance(dim, int) and dim > 0 for dim in state_size))):
            raise ValueError("State size must be a positive integer or a tuple of positive integers.")

        if not isinstance(action_size, int) or action_size <= 0:
            raise ValueError("Action size must be a positive integer.")

        self.buffer_size = buffer_size
        self.state_size = state_size if isinstance(state_size, tuple) else (state_size,)
        self.action_size = action_size
        self.sample_size = sample_size
        self.buffer = {
            'states': np.zeros((buffer_size, *self.state_size), dtype=np.float32),
            'actions': np.zeros((buffer_size, action_size), dtype=np.float32),
            'rewards': np.zeros(buffer_size, dtype=np.float32),
            'next_states': np.zeros((buffer_size, *self.state_size), dtype=np.float32),
            'dones': np.zeros(buffer_size, dtype=np.uint8)
        }
        self.ptr, self.size = 0, 0

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        If the buffer is full, the oldest experience is replaced.
        Add a new experience to the buffer.
        Parameters:
        state : np.array
            The state of the environment.
        action : np.array
            The action taken in the environment.
        reward : float
            The reward received from the environment.
        next_state : np.array
            The state of the environment after the action was taken.
        done : bool
            Whether the episode has ended.

        Raises:
        ValueError: If the dimensions of the state, action, or next_state do not match the expected dimensions.
        """

        # Check if state and next_state match the state_size dimension
        if not (isinstance(state, np.ndarray) and state.shape == self.state_size):
            raise ValueError(f"State dimension mismatch. Expected: {self.state_size}, Received: {state.shape}")

        if not (isinstance(next_state, np.ndarray) and next_state.shape == self.state_size):
            raise ValueError(f"Next state dimension mismatch. Expected: {self.state_size}, Received: {next_state.shape}")

        # Check if action matches the action_size dimension
        if not (isinstance(action, np.ndarray) and action.shape == (self.action_size,)):
            raise ValueError(f"Action dimension mismatch. Expected: {self.action_size}, Received: {action.shape}")

        # Check if reward is a float and done is a boolean
        if not isinstance(reward, float):
            raise ValueError(f"Reward should be a float. Received: {type(reward)}")

        if not isinstance(done, bool):
            raise ValueError(f"Done should be a boolean. Received: {type(done)}")
        idx = self.ptr % self.buffer_size
        self.buffer['states'][idx] = state
        self.buffer['actions'][idx] = action
        self.buffer['rewards'][idx] = reward
        self.buffer['next_states'][idx] = next_state
        self.buffer['dones'][idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, tensor=False, sample_size = None):
        """
        Sample a batch of experiences from the buffer.

        Parameters:
        tensor : bool
            If True, returns the sampled experiences as TensorFlow tensors.
            Otherwise, returns them as NumPy arrays.

        Returns:
        dict or str
            A dictionary containing states, actions, rewards, next_states, and dones for the sampled experiences, 
            or a string message if there is not enough data to sample.
        """
        if(sample_size is None):
            sample_size = self.sample_size
        if self.size < self.sample_size:
            raise ValueError(f"Not enough data in buffer to sample. Buffer size: {self.size}, Required: {self.sample_size}")
        #give me sample_size amount of random indexs
        idxs = np.random.choice(self.size, sample_size, replace=False)

        if tensor:
            return {key: tf.convert_to_tensor(self.buffer[key][idxs]) for key in self.buffer}
        else:
            return {key: self.buffer[key][idxs] for key in self.buffer}



    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return self.size

    def __str__(self):
        """
        Return a string representation of the ReplayBuffer object.
        """
        return (f"ReplayBuffer(\n"
                f"  Buffer Size: {self.buffer_size}\n"
                f"  State Size: {self.state_size}\n"
                f"  Action Size: {self.action_size}\n"
                f"  Sample Size: {self.sample_size}\n"
                f"  Current Size: {self.size}\n"
                f"  Current Pointer Position: {self.ptr}\n"
                f")")
