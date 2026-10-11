"""
Parameter sweep engine using Latin Hypercube Sampling.

Generates a space-filling experimental design and runs the cycle model
for each sample point. Results saved to parquet/csv.
"""

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from joblib import Parallel, delayed
import time
import os

from src.cycle_model import CycleConfig, run_single_evaluation
from src.adsorption import DEFAULTS_HCHO, DEFAULTS_CH3CHO


# (name, low, high, config_attribute, scale)
SWEEP_PARAMS = [
    ('T_exhaust',     473.15, 673.15, 'T_exhaust',     'linear'),
    ('u',             0.01,   0.15,   'u',             'linear'),
    ('L',             0.02,   0.10,   'L',             'linear'),
    ('d_bed',         0.02,   0.06,   'd_bed',         'linear'),
    ('d_p',           0.5e-3, 4.0e-3, 'd_p',           'linear'),
    ('eps',           0.30,   0.50,   'eps',           'linear'),
    ('C_in_hcho',     0.01,   0.10,   'C_in_hcho',     'linear'),
    ('C_in_ch3cho',   0.005,  0.05,   'C_in_ch3cho',   'linear'),
    ('t_adsorption',  300.0,  1200.0, 't_adsorption',  'linear'),
    ('t_desorption',  120.0,  600.0,  't_desorption',  'linear'),
    ('h_cold',        10.0,   50.0,   'h_cold',        'linear'),
    ('E_a_des_hcho',  50e3,   80e3,   None,            'linear'),
    ('k_LDF_hcho',    0.005,  0.02,   None,            'linear'),
]

N_PARAMS = len(SWEEP_PARAMS)


def generate_lhs_samples(n_samples, seed=42):
    """Generate Latin Hypercube samples scaled to parameter ranges."""
    np.random.seed(seed)
    lhs_unit = lhs(N_PARAMS, samples=n_samples, criterion='maximin')

    samples = np.zeros_like(lhs_unit)
    param_names = []

    for i, (name, lo, hi, _, scale) in enumerate(SWEEP_PARAMS):
        if scale == 'log':
            samples[:, i] = np.exp(np.log(lo) + lhs_unit[:, i] * (np.log(hi) - np.log(lo)))
        else:
            samples[:, i] = lo + lhs_unit[:, i] * (hi - lo)
        param_names.append(name)

    return samples, param_names


def sample_to_config(sample):
    """Convert a single LHS sample vector to a CycleConfig."""
    kwargs = {}
    params_hcho = DEFAULTS_HCHO.copy()
    params_ch3cho = DEFAULTS_CH3CHO.copy()

    for i, (name, _, _, attr, _) in enumerate(SWEEP_PARAMS):
        val = sample[i]
        if attr is not None:
            kwargs[attr] = val
        elif name == 'E_a_des_hcho':
            params_hcho['E_a_des'] = val
            params_ch3cho['E_a_des'] = val * 0.89
        elif name == 'k_LDF_hcho':
            params_hcho['k_LDF'] = val
            params_ch3cho['k_LDF'] = val * 0.8

    kwargs['params_hcho'] = params_hcho
    kwargs['params_ch3cho'] = params_ch3cho
    kwargs['Nz'] = 30
    kwargs['Nt_per_phase'] = 60

    return CycleConfig(**kwargs)


def run_sweep(n_samples=2000, n_jobs=4, seed=42, output_dir='data/generated'):
    """
    Run the full parameter sweep.

    Parameters
    ----------
    n_samples : int
    n_jobs : int
    seed : int
    output_dir : str
    """
    print(f"Generating {n_samples} LHS samples...")
    samples, param_names = generate_lhs_samples(n_samples, seed)

    print(f"Running {n_samples} simulations with {n_jobs} workers...")
    t0 = time.time()

    def eval_sample(i):
        config = sample_to_config(samples[i])
        result = run_single_evaluation(config)
        result['sample_id'] = i
        return result

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(eval_sample)(i) for i in range(n_samples)
    )

    elapsed = time.time() - t0
    print(f"Completed {n_samples} evaluations in {elapsed:.1f}s ({elapsed/n_samples:.3f}s each)")

    df = pd.DataFrame(results)

    for j, name in enumerate(param_names):
        df[f'input_{name}'] = samples[:, j]

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, 'full_parameter_sweep.parquet')
    csv_path = os.path.join(output_dir, 'full_parameter_sweep.csv')

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    successful = df[df['solver_success'] == True]
    print(f"\nSweep complete: {len(successful)}/{len(df)} successful")
    print(f"Saved to {parquet_path}")

    return df


if __name__ == '__main__':
    run_sweep(n_samples=2000, n_jobs=4)
