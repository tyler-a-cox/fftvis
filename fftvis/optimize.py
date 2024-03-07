import optax
import numpy as np
import functools

import tqdm
import jax
from jax import numpy as jnp
from .simulate import _FFT_simulator


def _compute_loss(labels: jnp.ndarray, predictions: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the loss between the labels and the predictions.

    Parameters:
    ----------
    labels : jnp.ndarray
        The true values.
    predictions : jnp.ndarray
        The predicted values.

    Returns:
    -------
    jnp.ndarray
        The loss.
    """
    # Initialize the model
    model = _FFT_simulator(
        times=times,
        freqs=freqs,
        ra=ra,
        dec=dec,
    )

    return optax.l2_loss(labels.real, predictions.real) + optax.l2_loss(
        labels.imag, predictions.imag
    )


def _evaluate_sky_model(
    model_parameters: dict, freqs: jnp.ndarray, pivot_freq: float = 150e6
):
    """
    Compute sky model from model parameters

    Parameters:
    ----------
    model_parameters : dict
        Model parameters for the sky model. Keys are the parameter names and values are the parameter values.
    freqs : jnp.ndarray
        Frequencies at which to evaluate the sky model.
    """
    return model_parameters["sky_model_amplitude"] * jnp.power(
        freqs / pivot_freq, model_parameters["spectral_index"]
    )


def fit_sky(
    model_parameters: dict,
    optimizer: optax.GradientTransformation,
    data: jnp.ndarray,
    times: jnp.ndarray,
    freqs: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    nsteps: int = 100,
) -> tuple[dict, jnp.ndarray]:
    """
    Fit a model to the sky. This function uses the jax library to fit the model to the data.

    Parameters:
    ----------
    model_parameters : dict
        A dictionary containing the model parameters.
    optimizer : optax.GradientTransformation
        The optimizer to use for the fitting.
    data : jnp.ndarray
        The data to fit. Data here has shape (ntimes, nfrequencies, nbaselines).
    times : jnp.ndarray
        The times of the data. Must match the number of times in data
    freqs : jnp.ndarray
        The frequencies of the data. Must match the number of frequencies in data
    baselines: jnp.ndarray
        The baselines of the data. Must match the number of baselines in data
    ra : jnp.ndarray
        Source positions in right ascension.
    dec : jnp.ndarray
        Source positions in declination.
    nsteps : int
        The number of steps to run the optimizer.

    Returns:
    -------
    dict
        The best fit model.
    jnp.ndarray
        The loss history.
    """
    opt_state = optimizer.init(model_parameters)
    losses = []

    @functools.partial(jax.vmap, in_axes=(None, 0, 0), axis_name="num_device")
    def parallelized_updates(model_parameters, coords, v):
        pass

    @jax.jit
    def update(model_parameters):
        """ """
        pass

    for _ in tqdm.tqdm(range(nsteps)):
        loss, grads = jax.value_and_grad(_compute_loss)(parameters, coords, v)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        losses.append(loss)

    return parameters, jnp.array(losses)
