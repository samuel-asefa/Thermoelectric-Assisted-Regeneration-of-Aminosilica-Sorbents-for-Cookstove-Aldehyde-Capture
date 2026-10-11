"""
Integrated cyclic adsorption/regeneration simulator.

Couples transport, adsorption kinetics, thermal model, TEG output, and pressure drop.

Cycle phases:
  1. ADSORPTION  – gas flows through bed, aldehydes captured at ~ambient T
  2. HEATING     – TEG-powered heater raises bed temperature
  3. DESORPTION  – thermal regeneration releases aldehydes
  4. COOLING     – bed returns to ambient temperature
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.adsorption import (DEFAULTS_HCHO, DEFAULTS_CH3CHO,
                             langmuir_equilibrium, net_adsorption_rate,
                             heat_of_adsorption, R_GAS)
from src.transport import run_transport_simulation, compute_mass_balance
from src.thermal import ThermalResistanceNetwork, THERMAL_DEFAULTS
from src.thermoelectric import compute_teg_performance, TEG_DEFAULTS
from src.pressure_drop import compute_pressure_drop_full, PRESSURE_DROP_DEFAULTS


class CycleConfig:
    """Configuration for a single cyclic simulation."""

    def __init__(self, **kwargs):
        # Bed geometry
        self.L = kwargs.get('L', 0.05)             # m
        self.d_bed = kwargs.get('d_bed', 0.04)     # m
        self.eps = kwargs.get('eps', 0.40)
        self.rho_sorbent = kwargs.get('rho_sorbent', 600.0)  # kg/m³

        # Flow
        self.u = kwargs.get('u', 0.05)             # m/s
        self.D_ax = kwargs.get('D_ax', 1e-5)       # m²/s

        # Inlet concentrations [mol/m³]
        # ~1 ppm HCHO at 25°C ≈ 0.04 mol/m³
        self.C_in_hcho = kwargs.get('C_in_hcho', 0.04)
        self.C_in_ch3cho = kwargs.get('C_in_ch3cho', 0.02)

        # Sorbent parameters
        self.params_hcho = kwargs.get('params_hcho', DEFAULTS_HCHO.copy())
        self.params_ch3cho = kwargs.get('params_ch3cho', DEFAULTS_CH3CHO.copy())

        # Thermal
        self.T_amb = kwargs.get('T_amb', 298.15)
        self.T_exhaust = kwargs.get('T_exhaust', 573.15)  # K (300°C)
        self.thermal_params = kwargs.get('thermal_params', THERMAL_DEFAULTS.copy())

        # TEG
        self.teg_params = kwargs.get('teg_params', TEG_DEFAULTS.copy())
        self.R_load = kwargs.get('R_load', None)  # None = matched load

        # Pressure drop
        self.d_p = kwargs.get('d_p', 2.0e-3)    # m
        self.phi = kwargs.get('phi', 0.8)

        # TEG thermal network
        self.R_interface = kwargs.get('R_interface', 0.5)   # K/W
        self.K_TEG_thermal = kwargs.get('K_TEG_thermal', 1.2)  # W/K
        self.h_cold = kwargs.get('h_cold', 25.0)   # W/(m²·K)
        self.A_cold = kwargs.get('A_cold', 0.004)  # m²

        # Cycle timing [s]
        self.t_adsorption = kwargs.get('t_adsorption', 600.0)
        self.t_heating = kwargs.get('t_heating', 60.0)
        self.t_desorption = kwargs.get('t_desorption', 300.0)
        self.t_cooling = kwargs.get('t_cooling', 120.0)

        # Numerical
        self.Nz = kwargs.get('Nz', 50)
        self.Nt_per_phase = kwargs.get('Nt_per_phase', 100)

    @property
    def A_bed(self):
        return np.pi / 4 * self.d_bed**2

    @property
    def V_bed(self):
        return self.A_bed * self.L

    @property
    def m_sorbent(self):
        """Total sorbent mass [kg]"""
        return self.rho_sorbent * self.V_bed * (1 - self.eps)

    @property
    def t_cycle(self):
        return self.t_adsorption + self.t_heating + self.t_desorption + self.t_cooling


def run_single_evaluation(config):
    """
    Run a single adsorption phase + simplified regeneration assessment.

    Fast version used for parameter sweeps; returns key performance metrics
    without full cycling.

    Parameters
    ----------
    config : CycleConfig

    Returns
    -------
    dict with all performance metrics
    """
    try:
        # TEG steady-state
        thermal_net = ThermalResistanceNetwork(
            T_exhaust=config.T_exhaust,
            T_amb=config.T_amb,
            R_interface=config.R_interface,
            K_TEG=config.K_TEG_thermal,
            h_cold=config.h_cold,
            A_cold=config.A_cold,
        )
        ss = thermal_net.solve_steady_state()

        teg = compute_teg_performance(
            T_h=ss['T_hot_TEG'], T_c=ss['T_cold_TEG'],
            params=config.teg_params, R_load=config.R_load
        )

        # Adsorption phase
        t_span = (0, config.t_adsorption)
        t_eval = np.linspace(0, config.t_adsorption, config.Nt_per_phase)

        result = run_transport_simulation(
            Nz=config.Nz, L=config.L, u=config.u, D_ax=config.D_ax,
            eps=config.eps, rho_sorbent=config.rho_sorbent,
            params_hcho=config.params_hcho,
            params_ch3cho=config.params_ch3cho,
            C_in_hcho=config.C_in_hcho,
            C_in_ch3cho=config.C_in_ch3cho,
            t_span=t_span, t_eval=t_eval,
            phase='adsorption',
        )

        # Pressure drop
        pd_params = dict(
            mu=1.81e-5, rho_gas=1.184,
            d_p=config.d_p, eps=config.eps, phi=config.phi
        )
        pd = compute_pressure_drop_full(config.u, config.L, config.d_bed, pd_params)

        # Capture metrics
        cap_eff_hcho = float(np.mean(result['capture_eff_hcho']))
        cap_eff_ch3cho = float(np.mean(result['capture_eff_ch3cho']))

        bt_hcho = result['breakthrough_hcho']
        bt_ch3cho = result['breakthrough_ch3cho']

        t_bt_hcho = _find_breakthrough_time(result['t'], bt_hcho, threshold=0.05)
        t_bt_ch3cho = _find_breakthrough_time(result['t'], bt_ch3cho, threshold=0.05)

        q_avg_hcho = float(np.mean(result['q_hcho'][-1, :]))
        q_avg_ch3cho = float(np.mean(result['q_ch3cho'][-1, :]))
        cap_util_hcho = q_avg_hcho / config.params_hcho['q_max']
        cap_util_ch3cho = q_avg_ch3cho / config.params_ch3cho['q_max']

        # Regeneration estimate
        T_regen = config.T_amb + 80.0
        E_sensible = (config.rho_sorbent * config.V_bed *
                      config.thermal_params['c_p_bed'] * 80.0)

        P_teg = float(np.atleast_1d(teg['P_max'])[0]) if np.ndim(teg['P_max']) > 0 else float(teg['P_max'])
        P_heater = P_teg * config.thermal_params.get('heater_efficiency', 0.95)

        t_heat_est = E_sensible / max(P_heater, 1e-10)

        k_des_hcho = config.params_hcho['A_des'] * np.exp(
            -config.params_hcho['E_a_des'] / (R_GAS * T_regen))
        regen_frac_hcho = 1.0 - np.exp(-k_des_hcho * config.t_desorption)

        k_des_ch3cho = config.params_ch3cho['A_des'] * np.exp(
            -config.params_ch3cho['E_a_des'] / (R_GAS * T_regen))
        regen_frac_ch3cho = 1.0 - np.exp(-k_des_ch3cho * config.t_desorption)

        E_available = P_heater * (config.t_heating + config.t_desorption)
        energy_sufficient = E_available >= E_sensible

        return dict(
            L=config.L, d_bed=config.d_bed, eps=config.eps,
            u=config.u, d_p=config.d_p,
            C_in_hcho=config.C_in_hcho, C_in_ch3cho=config.C_in_ch3cho,
            T_exhaust=config.T_exhaust, T_amb=config.T_amb,
            t_adsorption=config.t_adsorption,
            t_desorption=config.t_desorption,

            dT_TEG=float(np.atleast_1d(teg['dT'])[0]) if np.ndim(teg['dT']) > 0 else float(teg['dT']),
            T_hot_TEG=float(ss['T_hot_TEG']),
            T_cold_TEG=float(ss['T_cold_TEG']),
            P_TEG=P_teg,
            V_oc=float(np.atleast_1d(teg['V_oc'])[0]) if np.ndim(teg['V_oc']) > 0 else float(teg['V_oc']),

            capture_eff_hcho=cap_eff_hcho,
            capture_eff_ch3cho=cap_eff_ch3cho,
            t_breakthrough_hcho=t_bt_hcho,
            t_breakthrough_ch3cho=t_bt_ch3cho,
            cap_util_hcho=cap_util_hcho,
            cap_util_ch3cho=cap_util_ch3cho,
            q_avg_hcho=q_avg_hcho,
            q_avg_ch3cho=q_avg_ch3cho,

            regen_frac_hcho=float(regen_frac_hcho),
            regen_frac_ch3cho=float(regen_frac_ch3cho),
            T_regen=T_regen,
            t_heat_est=t_heat_est,
            P_heater=P_heater,
            energy_sufficient=energy_sufficient,
            E_sensible=E_sensible,
            E_available=E_available,

            dP=float(pd['dP']),
            Re_p=float(pd['Re_p']),
            residence_time=float(pd['residence_time']),
            fan_power=float(pd['fan_power']),

            solver_success=result['solver_success'],
            failure_state='none',
        )

    except Exception as e:
        return dict(failure_state=str(e), solver_success=False)


def run_full_cycling(config, n_cycles=10):
    """
    Run full cyclic adsorption/regeneration simulation.

    Parameters
    ----------
    config : CycleConfig
    n_cycles : int

    Returns
    -------
    dict with full time histories
    """
    Nz = config.Nz

    C_hcho = np.zeros(Nz)
    C_ch3cho = np.zeros(Nz)
    q_hcho = np.zeros(Nz)
    q_ch3cho = np.zeros(Nz)
    T_bed = np.full(Nz, config.T_amb)

    thermal_net = ThermalResistanceNetwork(
        T_exhaust=config.T_exhaust, T_amb=config.T_amb,
        R_interface=config.R_interface, K_TEG=config.K_TEG_thermal,
        h_cold=config.h_cold, A_cold=config.A_cold,
    )
    ss = thermal_net.solve_steady_state()
    teg = compute_teg_performance(ss['T_hot_TEG'], ss['T_cold_TEG'],
                                   config.teg_params, config.R_load)
    P_teg = float(np.atleast_1d(teg['P_max'])[0]) if np.ndim(teg['P_max']) > 0 else float(teg['P_max'])
    P_heater = P_teg * config.thermal_params.get('heater_efficiency', 0.95)

    all_t, all_C_out_hcho, all_C_out_ch3cho = [], [], []
    all_q_mean_hcho, all_q_mean_ch3cho = [], []
    all_T_mean, all_P_teg, all_phase = [], [], []

    t_offset = 0.0
    cycle_metrics = []

    for cycle in range(n_cycles):
        # --- Adsorption ---
        t_eval_ads = np.linspace(0, config.t_adsorption, config.Nt_per_phase)
        result_ads = run_transport_simulation(
            Nz=Nz, L=config.L, u=config.u, D_ax=config.D_ax,
            eps=config.eps, rho_sorbent=config.rho_sorbent,
            params_hcho=config.params_hcho, params_ch3cho=config.params_ch3cho,
            C_in_hcho=config.C_in_hcho, C_in_ch3cho=config.C_in_ch3cho,
            t_span=(0, config.t_adsorption), t_eval=t_eval_ads,
            phase='adsorption',
            q0_hcho=q_hcho, q0_ch3cho=q_ch3cho,
            C0_hcho=C_hcho, C0_ch3cho=C_ch3cho,
        )

        all_t.extend(result_ads['t'] + t_offset)
        all_C_out_hcho.extend(result_ads['C_hcho'][:, -1])
        all_C_out_ch3cho.extend(result_ads['C_ch3cho'][:, -1])
        all_q_mean_hcho.extend(np.mean(result_ads['q_hcho'], axis=1))
        all_q_mean_ch3cho.extend(np.mean(result_ads['q_ch3cho'], axis=1))
        all_T_mean.extend(np.mean(T_bed) * np.ones(len(result_ads['t'])))
        all_P_teg.extend(P_teg * np.ones(len(result_ads['t'])))
        all_phase.extend(['adsorption'] * len(result_ads['t']))

        t_offset += config.t_adsorption

        q_hcho = result_ads['q_hcho'][-1, :].copy()
        q_ch3cho = result_ads['q_ch3cho'][-1, :].copy()
        C_hcho = result_ads['C_hcho'][-1, :].copy()
        C_ch3cho = result_ads['C_ch3cho'][-1, :].copy()

        q_before_regen_hcho = np.mean(q_hcho)
        q_before_regen_ch3cho = np.mean(q_ch3cho)

        # --- Heating + Desorption ---
        t_regen_total = config.t_heating + config.t_desorption
        Nt_regen = config.Nt_per_phase
        t_eval_regen = np.linspace(0, t_regen_total, Nt_regen)

        m_bed = config.rho_sorbent * config.V_bed
        c_p = config.thermal_params['c_p_bed']
        h_loss = config.thermal_params['h_loss']
        A_loss = config.thermal_params['A_loss_per_vol'] * config.V_bed

        def thermal_rhs(t, T_state):
            T_val = T_state[0]
            dTdt = (P_heater - h_loss * A_loss * (T_val - config.T_amb)) / (m_bed * c_p)
            return [dTdt]

        T_init = float(np.mean(T_bed))
        sol_thermal = solve_ivp(thermal_rhs, (0, t_regen_total), [T_init],
                                 t_eval=t_eval_regen, method='RK45')
        T_regen_history = sol_thermal.y[0, :]

        dt_regen = np.diff(t_eval_regen)
        for i, dt in enumerate(dt_regen):
            T_now = T_regen_history[i+1]
            k_des_h = config.params_hcho['A_des'] * np.exp(
                -config.params_hcho['E_a_des'] / (R_GAS * T_now))
            k_des_a = config.params_ch3cho['A_des'] * np.exp(
                -config.params_ch3cho['E_a_des'] / (R_GAS * T_now))
            q_hcho = np.maximum(q_hcho * np.exp(-k_des_h * dt), 0.0)
            q_ch3cho = np.maximum(q_ch3cho * np.exp(-k_des_a * dt), 0.0)

        all_t.extend(t_eval_regen + t_offset)
        all_C_out_hcho.extend(np.zeros(Nt_regen))  # no flow during regen
        all_C_out_ch3cho.extend(np.zeros(Nt_regen))
        all_q_mean_hcho.extend(
            np.linspace(q_before_regen_hcho, float(np.mean(q_hcho)), Nt_regen))
        all_q_mean_ch3cho.extend(
            np.linspace(q_before_regen_ch3cho, float(np.mean(q_ch3cho)), Nt_regen))
        all_T_mean.extend(T_regen_history)
        all_P_teg.extend(P_teg * np.ones(Nt_regen))
        all_phase.extend(['regeneration'] * Nt_regen)

        t_offset += t_regen_total

        # --- Cooling ---
        Nt_cool = config.Nt_per_phase // 2
        t_eval_cool = np.linspace(0, config.t_cooling, Nt_cool)

        def cooling_rhs(t, T_state):
            T_val = T_state[0]
            return [-h_loss * A_loss * (T_val - config.T_amb) / (m_bed * c_p)]

        T_after_regen = float(T_regen_history[-1])
        sol_cool = solve_ivp(cooling_rhs, (0, config.t_cooling), [T_after_regen],
                              t_eval=t_eval_cool, method='RK45')
        T_cool_history = sol_cool.y[0, :]

        all_t.extend(t_eval_cool + t_offset)
        all_C_out_hcho.extend(np.zeros(Nt_cool))
        all_C_out_ch3cho.extend(np.zeros(Nt_cool))
        all_q_mean_hcho.extend(float(np.mean(q_hcho)) * np.ones(Nt_cool))
        all_q_mean_ch3cho.extend(float(np.mean(q_ch3cho)) * np.ones(Nt_cool))
        all_T_mean.extend(T_cool_history)
        all_P_teg.extend(np.zeros(Nt_cool))
        all_phase.extend(['cooling'] * Nt_cool)

        t_offset += config.t_cooling

        T_bed = np.full(Nz, float(T_cool_history[-1]))
        C_hcho = np.zeros(Nz)
        C_ch3cho = np.zeros(Nz)

        q_after_regen_hcho = float(np.mean(q_hcho))
        q_after_regen_ch3cho = float(np.mean(q_ch3cho))

        regen_completeness_hcho = 1.0 - q_after_regen_hcho / max(q_before_regen_hcho, 1e-30)
        regen_completeness_ch3cho = 1.0 - q_after_regen_ch3cho / max(q_before_regen_ch3cho, 1e-30)

        cycle_metrics.append(dict(
            cycle=cycle + 1,
            capture_eff_hcho=float(np.mean(result_ads['capture_eff_hcho'])),
            capture_eff_ch3cho=float(np.mean(result_ads['capture_eff_ch3cho'])),
            q_before_hcho=q_before_regen_hcho,
            q_after_hcho=q_after_regen_hcho,
            regen_completeness_hcho=regen_completeness_hcho,
            q_before_ch3cho=q_before_regen_ch3cho,
            q_after_ch3cho=q_after_regen_ch3cho,
            regen_completeness_ch3cho=regen_completeness_ch3cho,
            T_max_regen=float(np.max(T_regen_history)),
            P_TEG=P_teg,
        ))

    return dict(
        t=np.array(all_t),
        C_out_hcho=np.array(all_C_out_hcho),
        C_out_ch3cho=np.array(all_C_out_ch3cho),
        q_mean_hcho=np.array(all_q_mean_hcho),
        q_mean_ch3cho=np.array(all_q_mean_ch3cho),
        T_mean=np.array(all_T_mean),
        P_teg=np.array(all_P_teg),
        phase=all_phase,
        cycle_metrics=cycle_metrics,
        config=config,
    )


def _find_breakthrough_time(t, bt_curve, threshold=0.05):
    """Find time when breakthrough curve exceeds threshold."""
    idx = np.where(bt_curve > threshold)[0]
    if len(idx) == 0:
        return t[-1]
    return float(t[idx[0]])
