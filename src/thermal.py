"""
Transient thermal model for sorbent bed + TEG system.

Thermal network:
  Exhaust → [R_interface] → TEG hot side → [TEG] → TEG cold side → [R_sink] → Ambient
  TEG electrical output → Resistive heater → Sorbent bed

1D energy equation for sorbent bed:
  rho_bed * c_p * dT/dt = k_eff * d²T/dz² + Q_heater/V_bed - h_loss * (T - T_amb)
"""

import numpy as np
from scipy.integrate import solve_ivp


# Thermal parameters (provenance labels in parameter_provenance.csv)
THERMAL_DEFAULTS = dict(
    rho_bed=600.0,        # kg/m³  [ASSUMED]
    c_p_bed=1000.0,       # J/(kg·K)  [LITERATURE - silica ~800-1000]
    k_eff=0.2,            # W/(m·K)  [ASSUMED]
    h_loss=5.0,           # W/(m²·K)  [ASSUMED]
    A_loss_per_vol=40.0,  # m²/m³  [DERIVED]
    R_contact=0.01,       # K/W  [ASSUMED]
    heater_efficiency=0.95,  # [-]  [LITERATURE - resistive heaters ~95%]
    T_amb=298.15,         # K  [ASSUMED]
    T_exhaust=573.15,     # K  [LITERATURE - mid-range cookstove]
)


def build_thermal_rhs(Nz, dz, rho_bed, c_p, k_eff, h_loss, A_loss_per_vol,
                       T_amb, Q_heater_total, V_bed, phase='adsorption'):
    """
    Build RHS for 1D transient heat equation in sorbent bed.

    State: T(z) array of size Nz

    dT/dt = alpha * d²T/dz² + q_heater - h_loss_coeff * (T - T_amb)
    """
    alpha = k_eff / (rho_bed * c_p)
    loss_coeff = h_loss * A_loss_per_vol / (rho_bed * c_p)

    q_heater_vol = Q_heater_total / V_bed / (rho_bed * c_p) if phase == 'regeneration' else 0.0

    def rhs(t, T):
        d2T = np.zeros(Nz)
        d2T[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / dz**2
        d2T[0] = (T[1] - T[0]) / dz**2        # zero-gradient left BC
        d2T[-1] = (T[-2] - T[-1]) / dz**2     # zero-gradient right BC

        return alpha * d2T + q_heater_vol - loss_coeff * (T - T_amb)

    return rhs


def run_thermal_simulation(Nz, L, T0, t_span, t_eval,
                            Q_heater=0.0, phase='adsorption',
                            params=None):
    """
    Run thermal simulation for sorbent bed.

    Parameters
    ----------
    Nz : int
    L : float
        Bed length [m]
    T0 : float or ndarray
        Initial temperature [K]
    t_span : tuple
    t_eval : ndarray
    Q_heater : float
        Heater power [W]
    phase : str
    params : dict or None

    Returns
    -------
    dict with 't', 'z', 'T' (Nt, Nz), 'T_mean'
    """
    if params is None:
        params = THERMAL_DEFAULTS.copy()

    dz = L / Nz
    z = np.linspace(dz/2, L - dz/2, Nz)

    # Q_heater normalized per unit cross-sectional area; V_bed = L here
    V_bed = L

    T0_arr = np.full(Nz, T0) if np.isscalar(T0) else np.array(T0)

    rhs = build_thermal_rhs(Nz, dz,
                             params['rho_bed'], params['c_p_bed'], params['k_eff'],
                             params['h_loss'], params['A_loss_per_vol'],
                             params['T_amb'], Q_heater, V_bed, phase)

    sol = solve_ivp(rhs, t_span, T0_arr, method='BDF', t_eval=t_eval,
                    rtol=1e-8, atol=1e-10)

    T_field = sol.y.T  # (Nt, Nz)

    return dict(t=sol.t, z=z, T=T_field, T_mean=np.mean(T_field, axis=1), success=sol.success)


class ThermalResistanceNetwork:
    """
    Lumped-parameter thermal resistance network for TEG system.

    Nodes: exhaust → TEG hot side → TEG cold side → ambient.
    """

    def __init__(self, T_exhaust=573.15, T_amb=298.15,
                 R_interface=0.5, K_TEG=1.2, R_sink=2.0,
                 h_cold=25.0, A_cold=0.004):
        """
        Parameters
        ----------
        T_exhaust : float [K]
        T_amb : float [K]
        R_interface : float
            Thermal resistance, exhaust to TEG hot side [K/W]
        K_TEG : float [W/K]
        R_sink : float [K/W]
        h_cold : float [W/(m²·K)]
        A_cold : float [m²]
        """
        self.T_exhaust = T_exhaust
        self.T_amb = T_amb
        self.R_interface = R_interface
        self.K_TEG = K_TEG
        self.R_sink = R_sink
        self.h_cold = h_cold
        self.A_cold = A_cold

    def solve_steady_state(self):
        """
        Solve for steady-state TEG temperatures.

        Simplified pure-resistance model:
          R_total = R_interface + 1/K_TEG + R_cold
          Q = (T_exhaust - T_amb) / R_total
        """
        R_cold = 1.0 / (self.h_cold * self.A_cold) if self.h_cold * self.A_cold > 0 else self.R_sink
        R_total = self.R_interface + 1.0/self.K_TEG + R_cold

        Q_total = (self.T_exhaust - self.T_amb) / R_total
        T_hot_TEG = self.T_exhaust - Q_total * self.R_interface
        T_cold_TEG = T_hot_TEG - Q_total / self.K_TEG

        return dict(
            T_hot_TEG=T_hot_TEG,
            T_cold_TEG=T_cold_TEG,
            dT_TEG=T_hot_TEG - T_cold_TEG,
            Q_total=Q_total,
            R_total=R_total,
        )


def compute_energy_balance(T_field, t, rho_bed, c_p, V_bed, Q_heater, h_loss, A_loss, T_amb):
    """
    Check energy conservation for the thermal simulation.

    Returns
    -------
    dict with energy_stored, energy_in, energy_lost, residual
    """
    T_mean = np.mean(T_field, axis=1)
    energy_stored = rho_bed * c_p * V_bed * (T_mean[-1] - T_mean[0])
    energy_in = Q_heater * (t[-1] - t[0])
    Q_loss_t = h_loss * A_loss * (T_mean - T_amb)
    energy_lost = np.trapezoid(Q_loss_t, t)

    residual = energy_in - energy_stored - energy_lost
    rel_residual = abs(residual) / max(abs(energy_in), 1e-30)

    return dict(
        energy_stored=energy_stored,
        energy_in=energy_in,
        energy_lost=energy_lost,
        residual=residual,
        relative_residual=rel_residual,
    )
