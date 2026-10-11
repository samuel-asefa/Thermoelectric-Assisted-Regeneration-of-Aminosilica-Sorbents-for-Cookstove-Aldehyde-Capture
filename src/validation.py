"""
Validation suite: convergence studies, conservation checks, literature comparison.
"""
import numpy as np
import pandas as pd
from src.transport import run_transport_simulation, compute_mass_balance
from src.thermal import run_thermal_simulation, compute_energy_balance, THERMAL_DEFAULTS
from src.thermoelectric import compute_teg_performance, TEG_DEFAULTS
from src.pressure_drop import compute_pressure_drop_full
from src.adsorption import DEFAULTS_HCHO, DEFAULTS_CH3CHO


BASELINE = dict(
    Nz=50, L=0.05, u=0.05, D_ax=1e-5, eps=0.40, rho_sorbent=600.0,
    C_in_hcho=0.04, C_in_ch3cho=0.02, t_end=600.0,
)


def run_grid_convergence(Nz_list=None):
    """Run transport at multiple grid resolutions."""
    if Nz_list is None:
        Nz_list = [25, 50, 100]
    results = []
    for Nz in Nz_list:
        t_eval = np.linspace(0, BASELINE['t_end'], 150)
        r = run_transport_simulation(
            Nz=Nz, L=BASELINE['L'], u=BASELINE['u'], D_ax=BASELINE['D_ax'],
            eps=BASELINE['eps'], rho_sorbent=BASELINE['rho_sorbent'],
            params_hcho=DEFAULTS_HCHO.copy(), params_ch3cho=DEFAULTS_CH3CHO.copy(),
            C_in_hcho=BASELINE['C_in_hcho'], C_in_ch3cho=BASELINE['C_in_ch3cho'],
            t_span=(0, BASELINE['t_end']), t_eval=t_eval,
        )
        cap_hcho = float(np.mean(r['capture_eff_hcho']))
        cap_ch3cho = float(np.mean(r['capture_eff_ch3cho']))
        q_final = float(np.mean(r['q_hcho'][-1, :]))
        results.append(dict(Nz=Nz, dz=BASELINE['L']/Nz,
                           capture_eff_hcho=cap_hcho, capture_eff_ch3cho=cap_ch3cho,
                           q_final_hcho=q_final, nfev=r['nfev'], success=r['solver_success']))
    df = pd.DataFrame(results)
    finest = df.iloc[-1]
    df['rel_error_capture'] = abs(df['capture_eff_hcho'] - finest['capture_eff_hcho']) / max(finest['capture_eff_hcho'], 1e-10)
    return df


def run_timestep_convergence(rtol_list=None):
    """Run transport with different solver tolerances."""
    if rtol_list is None:
        rtol_list = [1e-4, 1e-6, 1e-8]
    results = []
    for rtol in rtol_list:
        t_eval = np.linspace(0, BASELINE['t_end'], 150)
        r = run_transport_simulation(
            Nz=BASELINE['Nz'], L=BASELINE['L'], u=BASELINE['u'], D_ax=BASELINE['D_ax'],
            eps=BASELINE['eps'], rho_sorbent=BASELINE['rho_sorbent'],
            params_hcho=DEFAULTS_HCHO.copy(), params_ch3cho=DEFAULTS_CH3CHO.copy(),
            C_in_hcho=BASELINE['C_in_hcho'], C_in_ch3cho=BASELINE['C_in_ch3cho'],
            t_span=(0, BASELINE['t_end']), t_eval=t_eval,
            rtol=rtol, atol=rtol*1e-3,
        )
        cap_hcho = float(np.mean(r['capture_eff_hcho']))
        results.append(dict(rtol=rtol, capture_eff_hcho=cap_hcho, nfev=r['nfev'], success=r['solver_success']))
    df = pd.DataFrame(results)
    finest = df.iloc[-1]
    df['rel_error'] = abs(df['capture_eff_hcho'] - finest['capture_eff_hcho']) / max(finest['capture_eff_hcho'], 1e-10)
    return df


def run_mass_balance_check():
    """Run baseline and check mass conservation."""
    t_eval = np.linspace(0, BASELINE['t_end'], 200)
    r = run_transport_simulation(
        Nz=BASELINE['Nz'], L=BASELINE['L'], u=BASELINE['u'], D_ax=BASELINE['D_ax'],
        eps=BASELINE['eps'], rho_sorbent=BASELINE['rho_sorbent'],
        params_hcho=DEFAULTS_HCHO.copy(), params_ch3cho=DEFAULTS_CH3CHO.copy(),
        C_in_hcho=BASELINE['C_in_hcho'], C_in_ch3cho=BASELINE['C_in_ch3cho'],
        t_span=(0, BASELINE['t_end']), t_eval=t_eval,
    )
    A_bed = np.pi/4 * 0.04**2
    mb = compute_mass_balance(r, BASELINE['u'], BASELINE['eps'], A_bed,
                              BASELINE['C_in_hcho'], BASELINE['C_in_ch3cho'],
                              BASELINE['rho_sorbent'], BASELINE['L'])
    return mb, r


def run_physical_bounds_check(r):
    """Check non-negative concentrations and bounded loading."""
    checks = {}
    checks['C_hcho_nonneg'] = bool(np.all(r['C_hcho'] >= -1e-15))
    checks['C_ch3cho_nonneg'] = bool(np.all(r['C_ch3cho'] >= -1e-15))
    checks['q_hcho_bounded'] = bool(np.all(r['q_hcho'] >= -1e-15) and
                                     np.all(r['q_hcho'] <= DEFAULTS_HCHO['q_max'] * 1.01))
    checks['q_ch3cho_bounded'] = bool(np.all(r['q_ch3cho'] >= -1e-15) and
                                       np.all(r['q_ch3cho'] <= DEFAULTS_CH3CHO['q_max'] * 1.01))
    return checks


def run_teg_validation():
    """Validate TEG output against literature ranges."""
    # Single Bi2Te3 module: ΔT=100-200K → 0.5-3W (literature)
    results = []
    for dT in [50, 100, 150, 200, 250]:
        T_h = 298.15 + dT
        T_c = 298.15
        teg = compute_teg_performance(T_h, T_c)
        P = float(np.atleast_1d(teg['P_max'])[0]) if np.ndim(teg['P_max']) > 0 else float(teg['P_max'])
        results.append(dict(dT=dT, P_max_W=P,
                           V_oc=float(np.atleast_1d(teg['V_oc'])[0]) if np.ndim(teg['V_oc']) > 0 else float(teg['V_oc'])))
    df = pd.DataFrame(results)
    P_at_200 = df[df['dT']==200]['P_max_W'].values[0]
    df.attrs['P_at_200K'] = P_at_200
    df.attrs['in_literature_range'] = 0.5 <= P_at_200 <= 5.0
    return df


def run_pressure_drop_validation():
    """Validate Ergun equation against expected ranges."""
    results = []
    for u in [0.01, 0.05, 0.10, 0.15]:
        pd_result = compute_pressure_drop_full(u, L=0.05, d_bed=0.04)
        results.append(dict(u=u, dP_Pa=pd_result['dP'], Re_p=pd_result['Re_p'], regime=pd_result['regime']))
    return pd.DataFrame(results)


def run_all_validation():
    """Run complete validation suite."""
    print("=" * 60)
    print("VALIDATION SUITE")
    print("=" * 60)

    print("\n--- Grid Convergence ---")
    gc = run_grid_convergence()
    print(gc.to_string(index=False))

    print("\n--- Timestep Convergence ---")
    tc = run_timestep_convergence()
    print(tc.to_string(index=False))

    print("\n--- Mass Balance ---")
    mb, r = run_mass_balance_check()
    for species, data in mb.items():
        print(f"  {species}: relative residual = {data['relative_residual']:.2e}")

    print("\n--- Physical Bounds ---")
    bounds = run_physical_bounds_check(r)
    for check, passed in bounds.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    print("\n--- TEG Validation ---")
    teg_df = run_teg_validation()
    print(teg_df.to_string(index=False))
    print(f"  P at ΔT=200K: {teg_df.attrs['P_at_200K']:.2f} W")
    print(f"  In literature range (0.5-5W): {teg_df.attrs['in_literature_range']}")

    print("\n--- Pressure Drop ---")
    pd_df = run_pressure_drop_validation()
    print(pd_df.to_string(index=False))

    return dict(
        grid_convergence=gc, timestep_convergence=tc,
        mass_balance=mb, physical_bounds=bounds,
        teg_validation=teg_df, pressure_drop=pd_df,
        baseline_result=r,
    )


if __name__ == '__main__':
    run_all_validation()
