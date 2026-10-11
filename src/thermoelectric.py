"""
Thermoelectric generator (TEG) model for Bi2Te3 modules.

Coupled thermal-electrical equations:
  V_oc = N * alpha * dT
  P_max = V_oc^2 / (4 * R_int)
  Q_h = N*alpha*T_h*I + K*dT - 0.5*I^2*R_int
  Q_c = N*alpha*T_c*I + K*dT + 0.5*I^2*R_int
"""

import numpy as np


# Default parameters for a typical Bi2Te3 module [LITERATURE]
TEG_DEFAULTS = dict(
    N_couples=127,           # number of thermoelectric couples
    alpha=220e-6,            # Seebeck coefficient per couple [V/K]  (200-240 µV/K)
    R_int=3.0,               # internal resistance [Ohm]
    K_TEG=1.2,               # thermal conductance [W/K]
    T_max_hot=453.15,        # max hot-side temperature [K] (180°C)
)


def compute_open_circuit_voltage(dT, N_couples, alpha):
    """
    Open-circuit voltage: V_oc = N * alpha * dT

    Parameters
    ----------
    dT : float or ndarray
        Temperature difference across TEG [K]
    N_couples : int
    alpha : float [V/K]

    Returns
    -------
    V_oc : float or ndarray [V]
    """
    return N_couples * alpha * dT


def compute_matched_load_power(dT, N_couples, alpha, R_int):
    """
    Maximum power at matched load (R_load = R_int).

    P_max = (N * alpha * dT)^2 / (4 * R_int)

    Parameters
    ----------
    dT : float or ndarray [K]
    N_couples : int
    alpha : float [V/K]
    R_int : float [Ohm]

    Returns
    -------
    P_max : float or ndarray [W]
    """
    V_oc = compute_open_circuit_voltage(dT, N_couples, alpha)
    return V_oc**2 / (4.0 * R_int)


def compute_load_power(dT, R_load, N_couples, alpha, R_int):
    """
    Power delivered to an arbitrary load resistance.

    I = V_oc / (R_int + R_load),  P = I^2 * R_load

    Parameters
    ----------
    dT : float or ndarray [K]
    R_load : float [Ohm]
    N_couples : int
    alpha : float [V/K]
    R_int : float [Ohm]

    Returns
    -------
    dict with V_oc, I, P_load, P_max
    """
    V_oc = compute_open_circuit_voltage(dT, N_couples, alpha)
    I = V_oc / (R_int + R_load)
    P_load = I**2 * R_load
    P_max = V_oc**2 / (4.0 * R_int)

    return dict(V_oc=V_oc, I=I, P_load=P_load, P_max=P_max, R_load=R_load)


def compute_teg_heat_flows(T_h, T_c, I, N_couples, alpha, R_int, K_TEG):
    """
    Heat flows through TEG module.

    Q_h = N*alpha*T_h*I + K*dT - 0.5*I^2*R_int
    Q_c = N*alpha*T_c*I + K*dT + 0.5*I^2*R_int

    Parameters
    ----------
    T_h, T_c : float
        Hot and cold side temperatures [K]
    I : float [A]
    N_couples : int
    alpha : float [V/K]
    R_int : float [Ohm]
    K_TEG : float [W/K]

    Returns
    -------
    dict with Q_h, Q_c, P_elec, eta
    """
    dT = T_h - T_c
    S_total = N_couples * alpha

    Q_h = S_total * T_h * I + K_TEG * dT - 0.5 * I**2 * R_int
    Q_c = S_total * T_c * I + K_TEG * dT + 0.5 * I**2 * R_int
    P_elec = Q_h - Q_c

    eta = P_elec / max(Q_h, 1e-10) if Q_h > 0 else 0.0

    return dict(Q_h=Q_h, Q_c=Q_c, P_elec=P_elec, eta=eta, dT=dT)


def compute_teg_performance(T_h, T_c, params=None, R_load=None):
    """
    Complete TEG performance calculation.

    Parameters
    ----------
    T_h : float or ndarray
        Hot-side temperature [K]
    T_c : float or ndarray
        Cold-side temperature [K]
    params : dict or None
    R_load : float or None
        Load resistance [Ohm]. If None, uses matched load.

    Returns
    -------
    dict with all TEG outputs
    """
    if params is None:
        params = TEG_DEFAULTS.copy()

    N = params['N_couples']
    alpha = params['alpha']
    R_int = params['R_int']
    K = params['K_TEG']

    dT = np.maximum(T_h - T_c, 0.0)

    if R_load is None:
        R_load = R_int  # matched load

    V_oc = compute_open_circuit_voltage(dT, N, alpha)
    I = V_oc / (R_int + R_load)
    P_load = I**2 * R_load
    P_max = V_oc**2 / (4.0 * R_int)

    S_total = N * alpha
    Q_h = S_total * T_h * I + K * dT - 0.5 * I**2 * R_int
    Q_c = S_total * T_c * I + K * dT + 0.5 * I**2 * R_int

    eta = np.where(Q_h > 0, P_load / Q_h, 0.0)

    return dict(
        T_h=T_h, T_c=T_c, dT=dT,
        V_oc=V_oc, I=I,
        P_load=P_load, P_max=P_max,
        Q_h=Q_h, Q_c=Q_c,
        eta=eta,
        R_load=R_load, R_int=R_int,
    )
