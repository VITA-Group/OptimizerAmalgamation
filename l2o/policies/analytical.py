"""Hand-crafted Optimizers."""

import tensorflow as tf

from .architectures import BaseCoordinateWisePolicy, BaseHierarchicalPolicy
from .moments import rms_momentum, rms_scaling


def _trainable(value, trainable=False):
    """Helper to optionally create trainable variable."""
    if trainable:
        return tf.Variable(
            tf.constant(value), trainable=True, dtype=tf.float32)
    else:
        return value


class AdamOptimizer(BaseCoordinateWisePolicy):
    """Adam Optimizer."""

    def init_layers(
            self, learning_rate=0.001, beta_1=0.9,
            beta_2=0.999, epsilon=1e-07, trainable=False):
        """Save hyperparameters (Adam has no layers)."""
        self.learning_rate = _trainable(learning_rate, trainable)
        self.beta_1 = _trainable(beta_1, trainable)
        self.beta_2 = _trainable(beta_2, trainable)
        self.epsilon = epsilon

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)

        update = self.learning_rate * m_hat / tf.sqrt(v_hat + self.epsilon)
        return update, states_new

    def get_initial_state(self, var):
        """Get initial optimizer state as a dictionary."""
        return {
            "m": tf.zeros(tf.shape(var)),
            "v": tf.zeros(tf.shape(var))
        }


class RMSPropOptimizer(BaseCoordinateWisePolicy):
    """RMSProp Optimizer."""

    def init_layers(
            self, learning_rate=0.001, rho=0.9,
            epsilon=1e-07, trainable=False):
        """Save hyperparameters."""
        self.learning_rate = _trainable(learning_rate, trainable)
        self.rho = _trainable(rho, trainable)
        self.epsilon = epsilon

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        scaled, states_new = rms_scaling(
            inputs, self.rho, states, epsilon=self.epsilon)

        return scaled * self.learning_rate, states_new

    def get_initial_state(self, var):
        """Get initial optimizer state as a dictionary."""
        return tf.zeros(tf.shape(var))


class MomentumOptimizer(BaseCoordinateWisePolicy):
    """Momentum Optimizer."""

    def init_layers(self, learning_rate=0.001, beta_1=0.9, trainable=False):
        """Save hyperparameters."""
        self.learning_rate = _trainable(learning_rate, trainable)
        self.beta_1 = _trainable(beta_1, trainable)

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        states_new = states * self.beta_1 + inputs * (1 - self.beta_1)

        return self.learning_rate * states_new, states_new

    def get_initial_state(self, var):
        """Get initial optimizer state as a dictionary."""
        return tf.zeros(tf.shape(var))


class PowerSignOptimizer(AdamOptimizer):
    """PowerSign optimizer (first variant), modified to be differentiable."""

    def init_layers(self, temperature=1.0, trainable=False, **kwargs):
        """Save hyperparameters."""
        self.temperature = _trainable(temperature, trainable)
        super().init_layers(**kwargs, trainable=trainable)

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)

        v_hat = tf.sqrt(states_new["v"] + self.epsilon)
        g_norm = inputs / v_hat
        m_norm = states_new["m"] / v_hat

        update = self.learning_rate * tf.exp(
            tf.math.tanh(g_norm) * tf.math.tanh(m_norm)) * inputs

        return update, states_new


class AddSignOptimizer(AdamOptimizer):
    """AddSign optimizer (first variant), modified to be differentiable."""

    def init_layers(self, temperature=1.0, trainable=False, **kwargs):
        """Save hyperparameters."""
        self.temperature = _trainable(temperature, trainable)
        super().init_layers(**kwargs, trainable=trainable)

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)

        v_hat = tf.sqrt(states_new["v"] + self.epsilon)
        g_norm = inputs / v_hat
        m_norm = states_new["m"] / v_hat

        update = self.learning_rate * (
            1 + tf.math.tanh(g_norm) * tf.math.tanh(m_norm)) * inputs

        return update, states_new


class AdaptivePowerSignOptimizer(AdamOptimizer):
    """Adaptive PowerSign variant, modified to be differentiable."""

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)

        v_hat = tf.sqrt(states_new["v"] + self.epsilon)
        g_norm = inputs / v_hat
        m_norm = states_new["m"] / v_hat

        update = self.learning_rate * tf.exp(
            tf.math.tanh(g_norm) * tf.math.tanh(m_norm)) * g_norm

        return update, states_new


class AdaptiveAddSignOptimizer(AdamOptimizer):
    """Adaptive PowerSign variant, modified to be differentiable."""

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)

        v_hat = tf.sqrt(states_new["v"] + self.epsilon)
        g_norm = inputs / v_hat
        m_norm = states_new["m"] / v_hat

        update = self.learning_rate * (
            1 + tf.math.tanh(g_norm) * tf.math.tanh(m_norm)) * g_norm

        return update, states_new


class SGDOptimizer(BaseCoordinateWisePolicy):
    """SGD Optimizer."""

    def init_layers(self, learning_rate=0.01, trainable=False):
        """Save hyperparameters."""
        self.learning_rate = _trainable(learning_rate, trainable)

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        # See BaseLearnToOptimizePolicy for why we need tf.constant(0.) instead
        # of None.
        return inputs * self.learning_rate, tf.constant(0.)

    def get_initial_state(self, var):
        """SGD has no state."""
        return tf.constant(0.)


class RectifiedAdamOptimizer(BaseHierarchicalPolicy):
    """Rectified Adam Optimizer."""

    def init_layers(
            self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
            epsilon=1e-07, weight_decay=0., total_steps=0.,
            warmup_proportion=0.1):
        """Save hyperparameters."""
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.rho_inf = 2 / (1 - beta_2) - 1
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.warmup_proportion = warmup_proportion

    def call(self, param, inputs, states, global_state, training=False):
        """Policy call override."""
        # Standard Adam update
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_hat = states_new["m"] / (1. - self.beta_1)

        # Length of approximated SMA
        beta_2_t = tf.math.pow(self.beta_2, global_state)
        rho_t = rho_inf - 2 * t * beta_2_t / (1 - beta_2_t)

        # Adaptive learning rate & rectification
        l_t = tf.math.sqrt((1 - beta_2_t) / states_new["v"])
        r_t = tf.math.sqrt(
            (rho_t - 4) * (rho_t - 2) * self.rho_inf
            / ((self.rho_inf - 4) * (self.rho_inf - 2))
        )

        # Variance is tractable check
        if rho_t > 4.0:
            update = self.learning_rate * r_t * m_hat * l_t
        else:
            update = self.learning_rate * m_hat

        return update, states_new

    def get_initial_state(self, var):
        """Get initial optimizer state as a dictionary."""
        return {
            "m": tf.zeros(tf.shape(var)),
            "v": tf.zeros(tf.shape(var))
        }

    def call_global(self, states, global_state, training=False):
        """Increment iteration count."""
        return global_state + 1

    def get_initial_state_global(self):
        """Initialize iteration count."""
        return tf.constant(0, dtype=tf.int64)
