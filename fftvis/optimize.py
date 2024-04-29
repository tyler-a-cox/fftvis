import optax
import numpy as np
import functools

import tqdm
import jax
from jax import numpy as jnp
from .simulate import _FFT_simulator
from jax._src.typing import Array


def _validate_input_model_parameters_sky(
    model_parameters: dict,
    times: Array,
    freqs: Array,
    ra: Array,
    dec: Array,
    use_point_source: bool,
    use_diffuse_component: bool,
    lmax: int,
):
    """
    Evaluate the input parameters for different fit modes.
    """
    # Check if the diffuse component is being used
    if use_diffuse_component:
        if "diffuse_component" not in model_parameters:
            raise ValueError(
                "Diffuse component is being used, but diffuse component parameters are not provided."
            )
        else:
            pass

    # Check for spectral index
    if "spectral_index" not in model_parameters:
        raise ValueError("Spectral index not provided.")
    else:
        spectral_index_shape = model_parameters["spectral_index"].shape
        if spectral_index_shape != ra.shape or spectral_index_shape != 1:
            raise ValueError(
                f"Spectral index shape {spectral_index_shape} does not match the number of sources {ra.shape} \
                  and is not a single value for the entire sky."
            )

    # Check for sky model amplitude
    if "sky_model_amplitude" not in model_parameters:
        raise ValueError("Sky model amplitude not provided.")
    else:
        sky_model_shape = model_parameters["sky_model_amplitude"].shape


def initialize_model_parameters_sky_fit(
    use_point_source: bool = False,
    sky_model_amplitude: Array = None,
    spectral_index: Array = None,
    use_diffuse_component: bool = False,
    diffuse_component: Array = None,
    diffuse_spectral_index: Array = None,
    lmax: int = None,
) -> dict:
    """
    Initialize the model parameters for fitting the sky.

    Parameters:
    ----------
    use_point_source : bool
        Whether to use a point source model.
    sky_model_amplitude : jnp.ndarray
        The amplitude of the sky model.
    spectral_index : jnp.ndarray
        The spectral index of the sky model.
    use_diffuse_component : bool
        Whether to use a diffuse component.
    diffuse_component : jnp.ndarray
        The diffuse component.
    diffuse_spectral_index : jnp.ndarray
        The spectral index of the diffuse component. Only used if use_diffuse_component is True.
    lmax : int
        The maximum spherical harmonic degree to use for the diffuse component.

    Returns:
    -------
    dict
        The model parameters.
    """
    model_parameters = {
        "spectral_index": spectral_index,
        "sky_model_amplitude": sky_model_amplitude,
    }

    if use_diffuse_component:
        model_parameters["diffuse_component"] = diffuse_component
        model_parameters["lmax"] = lmax

    return model_parameters


def initialize_model_parameters_beam_fit(
    beam_type: str = "airy",
    init_beam_diameter: float = 14.0,
    spectral_index: Array = 0.0,
    beam_vals: Array = None,
    u_coord: Array = None,
    v_coord: Array = None,
    beam_az: Array = None,
    beam_za: Array = None,
) -> dict:
    """
    Initialize the model parameters for fitting the beam.

    Parameters:
    ----------
    beam_type : str
        The type of beam to use. Options are "airy", "gaussian", "bessel_basis", and "uvbeam".
    init_beam_diameter : float
        The initial beam diameter in meters.
    spectral_index : jnp.ndarray
        The spectral index of the beam pattern.
    beam_vals : jnp.ndarray
        The beam values.
    u_coord : jnp.ndarray
        The u coordinates of the beam.
    v_coord : jnp.ndarray
        The v coordinates of the beam.
    beam_az : jnp.ndarray
        The azimuthal coordinates of the beam.
    beam_za : jnp.ndarray
        The zenith angle coordinates of the beam.

    Returns:
    -------
    model_parameters: dict
        Parameters for the beam model.
    """
    assert beam_type in [
        "airy",
        "gaussian",
        "bessel_basis",
        "uvbeam",
    ], "Beam type not recognized."

    if beam_type in ["airy", "gaussian"]:
        assert init_beam_diameter is not None, "Initial beam diameter not provided."

        # Pack the model parameters
        model_parameters = {
            "beam_diameter": init_beam_diameter,
        }

    elif beam_type == "uvbeam":
        model_parameters = {
            "beam_vals": beam_vals,
        }

    else:
        # Must be a bessel basis
        pass

    return model_parameters


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

    Parameters:
    ----------
    model_parameters : dict
        The model parameters to fit.
    fit_mode : str
        The fit mode to use.
    freqs : jnp.ndarray
        The frequencies of the data.
    times : jnp.ndarray
        The times of the data.
    ra : jnp.ndarray
        The right ascension of the source.
    dec : jnp.ndarray
        The declination of the source.

    Returns:
    -------
    tuple
        The model parameters, frequencies, times, right ascension, and declination.
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
    """
    Validate the inputs for fitting the sky.
    """
    pass


def _compute_loss_beam_fit(
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
    data: Array,
    times: Array,
    freqs: Array,
    ra: Array,
    dec: Array,
    fit_beam: bool = False,
    beam: str = "airy",
    beam_diameter: float = 14.6,
    beam_vals: Array = None,
    beam_az: Array = None,
    beam_za: Array = None,
    use_diffuse_component: bool = False,
    lmax: int = 20,
    nsteps: int = 100,
) -> tuple[dict, jnp.ndarray]:
    """
    Fit a model to the sky. This function uses the jax library to fit the model to the data.
    Option available to link times

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
    fit_beam: bool = True
        pass
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
    vis_model: jnp.ndarray,
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
    vis_model : jnp.ndarray
        Zeroth order visibility model to fit.
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

    Has option to fit in uv-pixel space or in the Bessel basis
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
