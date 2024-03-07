import numpy as np

import jax
from jax import lax, config
from jax._src import dtypes
from jax import numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import promote_dtypes_inexact

config.update("jax_enable_x64", True)  # Use 64-bit precision

speed_of_light = 299792458.0  # m/s

# polynomial coefficients for J0
PP0 = jnp.array(
    [
        7.96936729297347051624e-4,
        8.28352392107440799803e-2,
        1.23953371646414299388e0,
        5.44725003058768775090e0,
        8.74716500199817011941e0,
        5.30324038235394892183e0,
        9.99999999999999997821e-1,
    ]
)
PQ0 = jnp.array(
    [
        9.24408810558863637013e-4,
        8.56288474354474431428e-2,
        1.25352743901058953537e0,
        5.47097740330417105182e0,
        8.76190883237069594232e0,
        5.30605288235394617618e0,
        1.00000000000000000218e0,
    ]
)

QP0 = jnp.array(
    [
        -1.13663838898469149931e-2,
        -1.28252718670509318512e0,
        -1.95539544257735972385e1,
        -9.32060152123768231369e1,
        -1.77681167980488050595e2,
        -1.47077505154951170175e2,
        -5.14105326766599330220e1,
        -6.05014350600728481186e0,
    ]
)
QQ0 = jnp.array(
    [
        1.0,
        6.43178256118178023184e1,
        8.56430025976980587198e2,
        3.88240183605401609683e3,
        7.24046774195652478189e3,
        5.93072701187316984827e3,
        2.06209331660327847417e3,
        2.42005740240291393179e2,
    ]
)

YP0 = jnp.array(
    [
        1.55924367855235737965e4,
        -1.46639295903971606143e7,
        5.43526477051876500413e9,
        -9.82136065717911466409e11,
        8.75906394395366999549e13,
        -3.46628303384729719441e15,
        4.42733268572569800351e16,
        -1.84950800436986690637e16,
    ]
)
YQ0 = jnp.array(
    [
        1.04128353664259848412e3,
        6.26107330137134956842e5,
        2.68919633393814121987e8,
        8.64002487103935000337e10,
        2.02979612750105546709e13,
        3.17157752842975028269e15,
        2.50596256172653059228e17,
    ]
)

DR10 = 5.78318596294678452118e0
DR20 = 3.04712623436620863991e1

RP0 = jnp.array(
    [
        -4.79443220978201773821e9,
        1.95617491946556577543e12,
        -2.49248344360967716204e14,
        9.70862251047306323952e15,
    ]
)
RQ0 = jnp.array(
    [
        1.0,
        4.99563147152651017219e2,
        1.73785401676374683123e5,
        4.84409658339962045305e7,
        1.11855537045356834862e10,
        2.11277520115489217587e12,
        3.10518229857422583814e14,
        3.18121955943204943306e16,
        1.71086294081043136091e18,
    ]
)

# J1
RP1 = jnp.array(
    [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
)
RQ1 = jnp.array(
    [
        1.0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
)

PP1 = jnp.array(
    [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
)
PQ1 = jnp.array(
    [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
)

QP1 = jnp.array(
    [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
)
QQ1 = jnp.array(
    [
        1.0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]
)

YP1 = jnp.array(
    [
        1.26320474790178026440e9,
        -6.47355876379160291031e11,
        1.14509511541823727583e14,
        -8.12770255501325109621e15,
        2.02439475713594898196e17,
        -7.78877196265950026825e17,
    ]
)
YQ1 = jnp.array(
    [
        5.94301592346128195359e2,
        2.35564092943068577943e5,
        7.34811944459721705660e7,
        1.87601316108706159478e10,
        3.88231277496238566008e12,
        6.20557727146953693363e14,
        6.87141087355300489866e16,
        3.97270608116560655612e18,
    ]
)

Z1 = 1.46819706421238932572e1
Z2 = 4.92184563216946036703e1
PIO4 = 0.78539816339744830962  # pi/4
THPIO4 = 2.35619449019234492885  # 3*pi/4
SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)


@jax.jit
def j0_small(x: ArrayLike) -> Array:
    """
    Implementation of J0 for x < 5
    """
    z = jnp.square(x)
    p = (z - DR10) * (z - DR20)
    p = p * jnp.polyval(RP0, z) / jnp.polyval(RQ0, z)
    return jnp.where(x < 1e-5, 1 - z / 4.0, p)


@jax.jit
def j0_large(x: ArrayLike) -> Array:
    """
    Implementation of J0 for x >= 5
    """

    w = 5.0 / x
    q = 25.0 / jnp.square(x)
    p = jnp.polyval(PP0, q) / jnp.polyval(PQ0, q)
    q = jnp.polyval(QP0, q) / jnp.polyval(QQ0, q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@jax.jit
def j0(z: ArrayLike) -> Array:
    """
    Bessel function of the first kind of order zero and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.

    Reference:
    Cephes Mathematical Library.

    Args:
        z: The sampling point(s) at which the Bessel function of the first kind are
        computed.

    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.asarray(z)
    (z,) = promote_dtypes_inexact(z)
    z_dtype = lax.dtype(z)

    if dtypes.issubdtype(z_dtype, complex):
        raise ValueError("complex input not supported.")

    return jnp.where(jnp.abs(z) < 5.0, j0_small(jnp.abs(z)), j0_large(jnp.abs(z)))


@jax.jit
def j1_small(x: ArrayLike) -> Array:
    """
    Implementation of J1 for x < 5
    """
    z = x * x
    w = jnp.polyval(RP1, z) / jnp.polyval(RQ1, z)
    w = w * x * (z - Z1) * (z - Z2)
    return w


@jax.jit
def j1_large(x: ArrayLike) -> Array:
    """
    Implementation of J1 for x > 5
    """
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(PP1, z) / jnp.polyval(PQ1, z)
    q = jnp.polyval(QP1, z) / jnp.polyval(QQ1, z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@jax.jit
def j1(z: ArrayLike) -> Array:
    """
    Bessel function of the first kind of order one and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.

    Reference:
    Cephes mathematical library.

    Args:
        x: The sampling point(s) at which the Bessel function of the first kind are
        computed.

    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """

    z = jnp.asarray(z)
    (z,) = promote_dtypes_inexact(z)
    z_dtype = lax.dtype(z)

    if dtypes.issubdtype(z_dtype, complex):
        raise ValueError("complex input not supported.")

    return jnp.sign(z) * jnp.where(
        jnp.abs(z) < 5.0, j1_small(jnp.abs(z)), j1_large(jnp.abs(z))
    )


def get_pos_reds(antpos, decimals=3, include_autos=True):
    """
    Figure out and return list of lists of redundant baseline pairs. This function is a modified version of the
    get_pos_reds function in redcal. It is used to calculate the redundant baseline groups from antenna positions
    rather than from a list of baselines. This is useful for simulating visibilities with fftvis.

    Parameters:
    ----------
        antpos: dict
            dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}.
        decimals: int, optional
            Number of decimal places to round to when determining redundant baselines. default is 3.
        include_autos: bool, optional
            if True, include autos in the list of pos_reds. default is False
    Returns:
    -------
        reds: list of lists of redundant tuples of antenna indices (no polarizations),
        sorted by index with the first index of the first baseline the lowest in the group.
    """
    # Create a dictionary of redundant baseline groups
    uv_to_red_key = {}
    reds = {}

    # Compute baseline lengths and round to specified precision
    baselines = np.round(
        [
            antpos[aj] - antpos[ai]
            for ai in antpos
            for aj in antpos
            if ai < aj or include_autos and ai == aj
        ],
        decimals,
    )

    ci = 0
    for ai in antpos:
        for aj in antpos:
            if ai < aj or include_autos and ai == aj:
                u, v, _ = baselines[ci]

                if (u, v) not in uv_to_red_key and (-u, -v) not in uv_to_red_key:
                    reds[(ai, aj)] = [(ai, aj)]
                    uv_to_red_key[(u, v)] = (ai, aj)
                elif (-u, -v) in uv_to_red_key:
                    reds[uv_to_red_key[(-u, -v)]].append((aj, ai))
                elif (u, v) in uv_to_red_key:
                    reds[uv_to_red_key[(u, v)]].append((ai, aj))

                ci += 1

    return [reds[k] for k in reds]
