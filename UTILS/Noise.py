import numpy as np

class OUActionNoise:
    """
    Ornstein-Uhlenbeck process based noise generator.

    This class generates noise from a stochastic process described by Ornstein-Uhlenbeck process,
    which is used to add noise to the actions for exploration in DDPG algorithm.

    Parameters:
    mu : np.array
        The mean towards which the noise is directed.
    sigma : float
        The scale of the noise.
    theta : float
        The rate at which the noise should revert to the mean.
    dt : float
        The timestep delta for each step of noise generation.
    x0 : np.array or None
        The initial value for the noise. If None, it starts at zero.

    Attributes:
    mu : np.array
        The mean towards which the noise is directed.
    sigma : float
        The scale of the noise.
    theta : float
        The rate at which the noise should revert to the mean.
    dt : float
        The timestep delta for each step of noise generation.
    x0 : np.array
        The initial value for the noise.
    x_prev : np.array
        The previous value of the noise.
    """

    def __init__(self, mu, sigma, theta, dt, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        #effectively resetting
        self.x_prev = self.x0 if x0 is not None else np.zeros_like(self.mu)

    def reset(self):
        """
        Reset the noise to its initial state.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        """
        Generate the next noise value.

        Returns:
        np.array
            The next noise value based on the Ornstein-Uhlenbeck process.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
