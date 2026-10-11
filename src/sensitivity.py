"""
Global sensitivity analysis using Sobol indices (SALib).
"""
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
from src.cycle_model import CycleConfig, run_single_evaluation
from src.adsorption import DEFAULTS_HCHO, DEFAULTS_CH3CHO

PROBLEM = {
    'num_vars': 8,
    'names': ['T_exhaust', 'u', 'L', 'd_p', 'eps', 'C_in_hcho', 't_adsorption', 'h_cold'],
    'bounds': [
        [473.15, 673.15],
        [0.01, 0.15],
        [0.02, 0.10],
        [0.5e-3, 4.0e-3],
        [0.30, 0.50],
        [0.01, 0.10],
        [300.0, 1200.0],
        [10.0, 50.0],
    ]
}

OUTPUTS = ['capture_eff_hcho', 'capture_eff_ch3cho', 'dP', 'P_TEG', 'regen_frac_hcho']


def run_sensitivity(N=128, seed=42):
    """
    Run Sobol sensitivity analysis.
    N: base sample size (total evals = N*(2D+2))
    """
    np.random.seed(seed)
    param_values = saltelli.sample(PROBLEM, N, calc_second_order=False)
    n_samples = param_values.shape[0]
    print(f"Sobol analysis: {n_samples} samples for {PROBLEM['num_vars']} parameters")

    results = {k: np.zeros(n_samples) for k in OUTPUTS}

    for i in range(n_samples):
        x = param_values[i]
        config = CycleConfig(
            T_exhaust=x[0], u=x[1], L=x[2], d_p=x[3], eps=x[4],
            C_in_hcho=x[5], C_in_ch3cho=x[5]*0.5, t_adsorption=x[6], h_cold=x[7],
            Nz=25, Nt_per_phase=50,
        )
        r = run_single_evaluation(config)
        for k in OUTPUTS:
            results[k][i] = r.get(k, np.nan) if isinstance(r.get(k), (int, float)) else np.nan
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_samples} complete")

    si_results = {}
    for output_name in OUTPUTS:
        Y = results[output_name]
        valid = ~np.isnan(Y)
        if valid.sum() < n_samples * 0.8:
            print(f"  Warning: {output_name} has {(~valid).sum()} NaN values")
            Y = np.nan_to_num(Y, nan=np.nanmedian(Y))
        try:
            Si = sobol.analyze(PROBLEM, Y, calc_second_order=False)
            si_results[output_name] = {
                'S1': Si['S1'], 'ST': Si['ST'],
                'S1_conf': Si['S1_conf'], 'ST_conf': Si['ST_conf'],
                'names': PROBLEM['names'],
            }
        except Exception as e:
            print(f"  Sobol analysis failed for {output_name}: {e}")

    return si_results, results, param_values


def sensitivity_to_dataframe(si_results):
    """Convert Sobol results to DataFrame for export."""
    rows = []
    for output, si in si_results.items():
        for i, name in enumerate(si['names']):
            rows.append(dict(
                output=output, parameter=name,
                S1=si['S1'][i], S1_conf=si['S1_conf'][i],
                ST=si['ST'][i], ST_conf=si['ST_conf'][i],
            ))
    return pd.DataFrame(rows)
