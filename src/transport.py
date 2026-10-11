"""
1D axial packed-bed reactive transport model.

Solves:  dC/dt + u * dC/dz = D_ax * d²C/dz² - rho_bed * (1-eps)/eps * dq/dt

Method of Lines with upwind advection, central diffusion.
Time integration via scipy.integrate.solve_ivp (BDF).
"""

import numpy as np
from scipy.integrate import solve_ivp
from src.adsorption import langmuir_equilibrium, adsorption_rate_ldf, net_adsorption_rate


def build_transport_rhs(Nz, dz, u, D_ax, eps, rho_sorbent, params_hcho, params_ch3cho,
                        C_in_hcho, C_in_ch3cho, T_bed_func=None, phase='adsorption'):
    """
    Build the RHS function for the method-of-lines PDE system.

    State vector layout: [C_hcho(0..Nz-1), C_ch3cho(0..Nz-1), q_hcho(0..Nz-1), q_ch3cho(0..Nz-1)]

    Parameters
    ----------
    Nz : int
        Number of spatial grid points
    dz : float
        Grid spacing [m]
    u : float
        Superficial velocity [m/s]
    D_ax : float
        Axial dispersion coefficient [m²/s]
    eps : float
        Bed porosity [-]
    rho_sorbent : float
        Sorbent bulk density [kg/m³]
    params_hcho, params_ch3cho : dict
        Adsorption parameters for each species
    C_in_hcho, C_in_ch3cho : float
        Inlet concentrations [mol/m³]
    T_bed_func : callable or None
        Function T_bed_func(t) returning bed temperature [K] array of size Nz.
        If None, uses 298.15 K.
    phase : str
        'adsorption' or 'regeneration'
    """
    rho_s_g = rho_sorbent * 1000.0  # kg/m³ → g/m³
    coupling = (1.0 - eps) / eps * rho_s_g

    def rhs(t, y):
        C_h = y[0:Nz].copy()
        C_a = y[Nz:2*Nz].copy()
        q_h = y[2*Nz:3*Nz].copy()
        q_a = y[3*Nz:4*Nz].copy()

        C_h = np.maximum(C_h, 0.0)
        C_a = np.maximum(C_a, 0.0)
        q_h = np.maximum(q_h, 0.0)
        q_a = np.maximum(q_a, 0.0)

        if T_bed_func is not None:
            T = T_bed_func(t)
            if np.isscalar(T):
                T = np.full(Nz, T)
        else:
            T = np.full(Nz, 298.15)

        dqh_dt = net_adsorption_rate(C_h, q_h, T, params_hcho, phase)
        dqa_dt = net_adsorption_rate(C_a, q_a, T, params_ch3cho, phase)

        # HCHO spatial terms: upwind advection, central diffusion
        dCh_dz = np.zeros(Nz)
        dCh_dz[0] = (C_h[0] - C_in_hcho) / dz
        dCh_dz[1:] = (C_h[1:] - C_h[:-1]) / dz

        d2Ch_dz2 = np.zeros(Nz)
        d2Ch_dz2[1:-1] = (C_h[2:] - 2*C_h[1:-1] + C_h[:-2]) / dz**2
        d2Ch_dz2[0] = (C_h[1] - 2*C_h[0] + C_in_hcho) / dz**2   # Danckwerts inlet
        d2Ch_dz2[-1] = (C_h[-2] - C_h[-1]) / dz**2               # zero-gradient outlet

        dCh_dt = -u * dCh_dz + D_ax * d2Ch_dz2 - coupling * dqh_dt

        # CH3CHO spatial terms
        dCa_dz = np.zeros(Nz)
        dCa_dz[0] = (C_a[0] - C_in_ch3cho) / dz
        dCa_dz[1:] = (C_a[1:] - C_a[:-1]) / dz

        d2Ca_dz2 = np.zeros(Nz)
        d2Ca_dz2[1:-1] = (C_a[2:] - 2*C_a[1:-1] + C_a[:-2]) / dz**2
        d2Ca_dz2[0] = (C_a[1] - 2*C_a[0] + C_in_ch3cho) / dz**2
        d2Ca_dz2[-1] = (C_a[-2] - C_a[-1]) / dz**2

        dCa_dt = -u * dCa_dz + D_ax * d2Ca_dz2 - coupling * dqa_dt

        return np.concatenate([dCh_dt, dCa_dt, dqh_dt, dqa_dt])

    return rhs


def run_transport_simulation(Nz, L, u, D_ax, eps, rho_sorbent,
                              params_hcho, params_ch3cho,
                              C_in_hcho, C_in_ch3cho,
                              t_span, t_eval=None,
                              T_bed_func=None, phase='adsorption',
                              q0_hcho=None, q0_ch3cho=None,
                              C0_hcho=None, C0_ch3cho=None,
                              method='BDF', rtol=1e-6, atol=1e-9):
    """
    Run a 1D packed-bed transport simulation.

    Returns
    -------
    result : dict with keys:
        't', 'z', 'C_hcho', 'C_ch3cho', 'q_hcho', 'q_ch3cho',
        'breakthrough_hcho', 'breakthrough_ch3cho',
        'capture_eff_hcho', 'capture_eff_ch3cho'
    """
    dz = L / Nz
    z = np.linspace(dz/2, L - dz/2, Nz)  # cell centers

    C0_hcho = np.zeros(Nz) if C0_hcho is None else C0_hcho
    C0_ch3cho = np.zeros(Nz) if C0_ch3cho is None else C0_ch3cho
    q0_hcho = np.zeros(Nz) if q0_hcho is None else q0_hcho
    q0_ch3cho = np.zeros(Nz) if q0_ch3cho is None else q0_ch3cho

    y0 = np.concatenate([C0_hcho, C0_ch3cho, q0_hcho, q0_ch3cho])

    rhs = build_transport_rhs(Nz, dz, u, D_ax, eps, rho_sorbent,
                               params_hcho, params_ch3cho,
                               C_in_hcho, C_in_ch3cho,
                               T_bed_func, phase)

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(rhs, t_span, y0, method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=t_span[1]/50)

    if not sol.success:
        print(f"Warning: solver did not converge: {sol.message}")

    C_hcho = sol.y[0:Nz, :].T
    C_ch3cho = sol.y[Nz:2*Nz, :].T
    q_hcho = sol.y[2*Nz:3*Nz, :].T
    q_ch3cho = sol.y[3*Nz:4*Nz, :].T

    bt_hcho = C_hcho[:, -1] / max(C_in_hcho, 1e-30)
    bt_ch3cho = C_ch3cho[:, -1] / max(C_in_ch3cho, 1e-30)

    return dict(
        t=sol.t, z=z,
        C_hcho=C_hcho, C_ch3cho=C_ch3cho,
        q_hcho=q_hcho, q_ch3cho=q_ch3cho,
        breakthrough_hcho=bt_hcho,
        breakthrough_ch3cho=bt_ch3cho,
        capture_eff_hcho=1.0 - np.clip(bt_hcho, 0, 1),
        capture_eff_ch3cho=1.0 - np.clip(bt_ch3cho, 0, 1),
        Nz=Nz, L=L, dz=dz,
        solver_success=sol.success,
        solver_message=sol.message,
        nfev=sol.nfev,
    )


def compute_mass_balance(result, u, eps, A_bed, C_in_hcho, C_in_ch3cho, rho_sorbent, L):
    """
    Check mass balance: moles in = moles adsorbed + moles out.

    Returns dict with residuals for each species.
    """
    t = result['t']
    Nz = result['Nz']
    dz = result['dz']

    balances = {}
    for species, C_in, C_field, q_field in [
        ('hcho', C_in_hcho, result['C_hcho'], result['q_hcho']),
        ('ch3cho', C_in_ch3cho, result['C_ch3cho'], result['q_ch3cho']),
    ]:
        Q_vol = u * A_bed * eps
        moles_in = Q_vol * C_in * (t[-1] - t[0])

        C_out = C_field[:, -1]
        moles_out = Q_vol * np.trapezoid(C_out, t)

        V_cell = A_bed * dz
        moles_gas = np.sum(C_field[-1, :]) * V_cell * eps

        m_sorbent_per_cell = rho_sorbent * 1000 * V_cell * (1 - eps)  # g
        moles_ads = np.sum(q_field[-1, :]) * m_sorbent_per_cell

        residual = moles_in - moles_out - moles_gas - moles_ads
        rel_residual = abs(residual) / max(moles_in, 1e-30)

        balances[species] = dict(
            moles_in=moles_in, moles_out=moles_out,
            moles_gas=moles_gas, moles_adsorbed=moles_ads,
            residual=residual, relative_residual=rel_residual
        )

    return balances
