"""
Adsorption equilibrium, kinetics, and thermal desorption models for
formaldehyde and acetaldehyde on amine-functionalized silica.
"""

import numpy as np

R_GAS = 8.314  # J/(mol·K)

# Langmuir isotherm parameters
# HCHO on primary-amine-functionalized mesoporous silica
# q_max ~ 1.4 mmol/g  (Srisuda & Virote 2008; Bollini et al. 2012)
DEFAULTS_HCHO = dict(
    q_max=1.4e-3,      # mol/g_sorbent  [LITERATURE]
    K_L=500.0,          # m³/mol         [ASSUMED - order consistent with strong chemisorption]
    k_LDF=0.01,         # 1/s            [ASSUMED - typical LDF for mesoporous sorbents]
    E_a_des=65.0e3,     # J/mol          [LITERATURE - chemisorption range 57-76 kJ/mol]
    A_des=1.0e6,        # 1/s            [ASSUMED]
    dH_ads=-60.0e3,     # J/mol          [ASSUMED - exothermic chemisorption]
)

# CH3CHO: slightly lower capacity and affinity than HCHO
DEFAULTS_CH3CHO = dict(
    q_max=1.0e-3,       # mol/g_sorbent  [ASSUMED - ~70% of HCHO capacity]
    K_L=350.0,          # m³/mol         [ASSUMED]
    k_LDF=0.008,        # 1/s            [ASSUMED]
    E_a_des=58.0e3,     # J/mol          [LITERATURE - lower end of chemisorption range]
    A_des=8.0e5,        # 1/s            [ASSUMED]
    dH_ads=-55.0e3,     # J/mol          [ASSUMED]
)


def langmuir_equilibrium(C, q_max, K_L):
    """
    Langmuir isotherm: q_eq = q_max * K_L * C / (1 + K_L * C)

    Parameters
    ----------
    C : float or ndarray
        Gas-phase concentration [mol/m³]
    q_max : float
        Maximum adsorption capacity [mol/g_sorbent]
    K_L : float
        Langmuir constant [m³/mol]

    Returns
    -------
    q_eq : float or ndarray
        Equilibrium loading [mol/g_sorbent]
    """
    C = np.maximum(C, 0.0)
    return q_max * K_L * C / (1.0 + K_L * C)


def adsorption_rate_ldf(q, q_eq, k_LDF):
    """
    Linear driving force (LDF) adsorption kinetics.

    dq/dt = k_LDF * (q_eq - q)

    Parameters
    ----------
    q : float or ndarray
        Current sorbent loading [mol/g_sorbent]
    q_eq : float or ndarray
        Equilibrium loading [mol/g_sorbent]
    k_LDF : float
        LDF mass transfer coefficient [1/s]

    Returns
    -------
    dqdt : float or ndarray
        Rate of change of loading [mol/(g_sorbent·s)]
    """
    return k_LDF * (q_eq - q)


def desorption_rate_arrhenius(q, T, E_a_des, A_des):
    """
    Temperature-dependent desorption rate (Arrhenius kinetics).

    dq/dt = -k_des * q,  k_des = A_des * exp(-E_a_des / (R * T))

    Parameters
    ----------
    q : float or ndarray
        Current sorbent loading [mol/g_sorbent]
    T : float or ndarray
        Sorbent temperature [K]
    E_a_des : float
        Desorption activation energy [J/mol]
    A_des : float
        Pre-exponential factor [1/s]

    Returns
    -------
    dqdt : float or ndarray
        Rate of change of loading [mol/(g_sorbent·s)] (negative)
    """
    k_des = A_des * np.exp(-E_a_des / (R_GAS * T))
    return -k_des * q


def net_adsorption_rate(C, q, T, params, phase='adsorption'):
    """
    Compute net rate of change of sorbent loading.

    Adsorption: LDF towards Langmuir equilibrium.
    Regeneration: Arrhenius desorption.

    Parameters
    ----------
    C : float or ndarray
        Gas-phase concentration [mol/m³]
    q : float or ndarray
        Current loading [mol/g_sorbent]
    T : float or ndarray
        Temperature [K]
    params : dict
        Sorbent parameters (q_max, K_L, k_LDF, E_a_des, A_des)
    phase : str
        'adsorption' or 'regeneration'

    Returns
    -------
    dqdt : float or ndarray
        Rate of loading change [mol/(g_sorbent·s)]
    """
    if phase == 'adsorption':
        q_eq = langmuir_equilibrium(C, params['q_max'], params['K_L'])
        dqdt = adsorption_rate_ldf(q, q_eq, params['k_LDF'])
    elif phase == 'regeneration':
        dqdt = desorption_rate_arrhenius(q, T, params['E_a_des'], params['A_des'])
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Enforce physical bounds
    dqdt = np.where(q <= 0, np.maximum(dqdt, 0.0), dqdt)
    dqdt = np.where(q >= params['q_max'], np.minimum(dqdt, 0.0), dqdt)

    return dqdt


def heat_of_adsorption(dqdt, dH_ads):
    """
    Heat source/sink from adsorption/desorption.

    Q_ads = -dH_ads * dqdt   (exothermic: dH < 0, dqdt > 0 → Q > 0)

    Parameters
    ----------
    dqdt : float or ndarray
        Rate of loading change [mol/(g_sorbent·s)]
    dH_ads : float
        Enthalpy of adsorption [J/mol] (negative for exothermic)

    Returns
    -------
    Q : float or ndarray
        Volumetric heat generation rate [W/g_sorbent]
    """
    return -dH_ads * dqdt
