import optax
import numpy as np
import functools

import tqdm
import jax
from jax import numpy as jnp
from .simulate import _FFT_simulator
from jax._src.typing import Array


#
def _evaluate_input_model_parameters(
    model_parameters: dict,
    fit_mode: str,
    freqs: Array,
    times: Array,
    ra: Array,
    dec: Array,
):
    """
    Evaluate the input parameters for different fit modes.
    """
    if fit_mode == "sky":
        return model_parameters, freqs, times, ra, dec
    elif fit_mode == "mutual_coupling":
        return model_parameters, freqs, times, ra, dec
    elif fit_mode == "beam":
        return model_parameters, freqs, times, ra, dec
    else:
        raise ValueError(f"Fit mode {fit_mode} not recognized.")


def _validate_sky_fit_inputs(beam: str, beam_diameter: float, beam_vals: Array):
    """ """
    pass


def _compute_loss_sky_fit(
    model_parameters: dict,
    freqs: Array,
    times: Array,
    ra: Array,
    dec: Array,
    labels: Array,
) -> jnp.ndarray:
    """
    Compute the loss between the labels and the predictions.

    Parameters:
    ----------
    model_parameters : dict
        The model parameters to fit.
    freqs : jnp.ndarray
        The frequencies of the data.
    times : jnp.ndarray
        The times of the data.
    ra : jnp.ndarray
        The right ascension of the source.
    dec : jnp.ndarray
        The declination of the source.
    labels : jnp.ndarray
        The data to fit.

    Returns:
    -------
    jnp.ndarray
        The loss.
    """
    # Evaluate the sky model for a given set of parameters
    sky_model = _evaluate_sky_model(model_parameters=model_parameters, freqs=freqs)

    # Initialize the model
    predictions = _FFT_simulator(
        sky_model=sky_model,
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


def _fit_sky(
    model_parameters: dict,
    optimizer: optax.GradientTransformation,
    data: jnp.ndarray,
    times: jnp.ndarray,
    freqs: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    beam: str = "airy",
    beam_diameter: float = 14.6,
    beam_vals: jnp.ndarray = None,
    beam_az: jnp.ndarray = None,
    beam_za: jnp.ndarray = None,
    diffuse_component: bool = False,
    lmax: int = 50,
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
        """
        Parallize data across devices to compute updates.
        """
        pass

    @jax.jit
    def update(model_parameters):
        """
        Update the model parameters.
        """
        pass

    for _ in tqdm.tqdm(range(nsteps)):
        # TODO: Add support for parallelized updates
        loss, grads = jax.value_and_grad(_compute_loss_sky_fit)(parameters, coords, v)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        losses.append(loss)

    return parameters, jnp.array(losses)


def _fit_mutual_coupling(
    model_parameters: dict,
    optimizer: optax.GradientTransformation,
    data: jnp.ndarray,
    times: jnp.ndarray,
    freqs: jnp.ndarray,
    sky_model: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    nsteps: int = 100,
) -> tuple[dict, jnp.ndarray]:
    """
    Skeleton function for fitting mutual coupling.

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
    sky_model : jnp.ndarray
        The sky model to use for the fitting.
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
    pass


def _fit_beam(
    model_parameters: dict,
    optimizer: optax.GradientTransformation,
    data: jnp.ndarray,
    times: jnp.ndarray,
    freqs: jnp.ndarray,
    ra: jnp.ndarray,
    dec: jnp.ndarray,
    sky_model: jnp.ndarray,
    nsteps: int = 100,
) -> tuple[dict, jnp.ndarray]:
    """
    Skeleton function for fitting the beam.
    """
    model_parameters = _evaluate_input_model_parameters(
        model_parameters=model_parameters,
        fit_mode="mutual_coupling",
        freqs=freqs,
        times=times,
        ra=ra,
        dec=dec,
    )

    # Initialize the model
    opt_state = optimizer.init(model_parameters)
    losses = []

    for _ in tqdm.tqdm(range(nsteps)):
        # TODO: Add support for parallelized updates
        loss, grads = jax.value_and_grad(_compute_loss_sky_fit)(parameters, coords, v)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        losses.append(loss)

    return parameters, jnp.array(losses)
