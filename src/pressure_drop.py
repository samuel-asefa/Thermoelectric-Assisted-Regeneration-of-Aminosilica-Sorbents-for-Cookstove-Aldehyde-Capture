"""
Packed-bed pressure drop model using the Ergun equation.

ΔP/L = 150·μ·(1-ε)²·u / (ε³·d_p²·φ²) + 1.75·ρ·(1-ε)·u² / (ε³·d_p·φ)
"""

import numpy as np


PRESSURE_DROP_DEFAULTS = dict(
    mu=1.81e-5,        # Pa·s, air at 25°C  [LITERATURE]
    rho_gas=1.184,     # kg/m³, air at 25°C  [LITERATURE]
    d_p=2.0e-3,        # m  [ASSUMED]
    phi=0.8,           # sphericity  [LITERATURE - 0.6-0.9]
    eps=0.40,          # bed porosity  [ASSUMED]
)


def ergun_pressure_drop(u, L, d_p, eps, phi, mu, rho_gas):
    """
    Ergun equation for packed-bed pressure drop.

    Parameters
    ----------
    u : float or ndarray
        Superficial velocity [m/s]
    L : float [m]
    d_p : float [m]
    eps : float [-]
    phi : float [-]
    mu : float [Pa·s]
    rho_gas : float [kg/m³]

    Returns
    -------
    dP : float or ndarray [Pa]
    """
    term_viscous = 150.0 * mu * (1 - eps)**2 * u / (eps**3 * (d_p * phi)**2)
    term_inertial = 1.75 * rho_gas * (1 - eps) * u**2 / (eps**3 * d_p * phi)
    return (term_viscous + term_inertial) * L


def ergun_pressure_gradient(u, d_p, eps, phi, mu, rho_gas):
    """Pressure gradient [Pa/m] from Ergun equation."""
    term_viscous = 150.0 * mu * (1 - eps)**2 * u / (eps**3 * (d_p * phi)**2)
    term_inertial = 1.75 * rho_gas * (1 - eps) * u**2 / (eps**3 * d_p * phi)
    return term_viscous + term_inertial


def compute_reynolds_particle(u, d_p, eps, rho_gas, mu):
    """
    Particle Reynolds number for packed bed.

    Re_p = rho * u * d_p / (mu * (1 - eps))
    """
    return rho_gas * u * d_p / (mu * (1 - eps))


def compute_pressure_drop_full(u, L, d_bed, params=None):
    """
    Full pressure drop calculation with diagnostics.

    Parameters
    ----------
    u : float [m/s]
    L : float [m]
    d_bed : float [m]
    params : dict or None

    Returns
    -------
    dict with dP, Re_p, regime, fan_power, residence_time
    """
    if params is None:
        params = PRESSURE_DROP_DEFAULTS.copy()

    d_p = params['d_p']
    eps = params['eps']
    phi = params['phi']
    mu = params['mu']
    rho = params['rho_gas']

    dP = ergun_pressure_drop(u, L, d_p, eps, phi, mu, rho)
    Re_p = compute_reynolds_particle(u, d_p, eps, rho, mu)

    if np.isscalar(Re_p):
        regime = 'viscous' if Re_p < 10 else ('inertial' if Re_p > 1000 else 'transitional')
    else:
        regime = np.where(Re_p < 10, 'viscous', np.where(Re_p > 1000, 'inertial', 'transitional'))

    A_bed = np.pi / 4 * d_bed**2
    Q_vol = u * A_bed
    fan_power = dP * Q_vol
    tau = L * eps / u if u > 0 else np.inf

    return dict(
        dP=dP,
        dP_per_m=dP/L,
        Re_p=Re_p,
        regime=regime,
        fan_power=fan_power,
        residence_time=tau,
        Q_vol=Q_vol,
    )
