import numpy as np
import tensorflow as tf
from Replay_Buffer import ReplayBuffer  # Assuming your class is saved in replay_buffer.py

def test_replay_buffer():
    # Create a ReplayBuffer instance
    buffer_size = 1000
    state_size = (4,)  # Example state size
    action_size = 2  # Example action size
    sample_size = 10
    replay_buffer = ReplayBuffer(buffer_size, state_size, action_size, sample_size)

    # Add experiences to the buffer
    for _ in range(50):  # Add fewer experiences than buffer_size
        state = np.random.randn(*state_size)
        action = np.random.randn(action_size)
        reward = np.random.rand()
        next_state = np.random.randn(*state_size)
        done = bool(np.random.choice([True, False]))
        replay_buffer.add(state, action, reward, next_state, done)

    # Test sample method
    sampled_experiences = replay_buffer.sample()
    assert isinstance(sampled_experiences, dict), "Sampled experiences should be a dictionary"
    assert all(len(sampled_experiences[key]) == sample_size for key in sampled_experiences), "Each sampled batch should have the correct sample size"

    # Test sample method with tensor conversion
    sampled_experiences_tensor = replay_buffer.sample(tensor=True)
    assert isinstance(sampled_experiences_tensor, dict), "Sampled experiences (tensor) should be a dictionary"
    assert all(isinstance(val, tf.Tensor) for val in sampled_experiences_tensor.values()), "All values in sampled experiences should be TensorFlow tensors"

    # Test the length of the buffer
    assert len(replay_buffer) == 50, "Buffer length should match the number of added experiences"

    print("All tests passed!")

test_replay_buffer()
