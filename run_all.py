#!/usr/bin/env python3
"""
Master runner: executes the complete computational study.

Phases:
  1. Validation
  2. Baseline spatiotemporal results
  3. Parameter sweep (LHS)
  4. Sensitivity analysis (Sobol)
  5. Cyclic simulations
  6. Surrogate model
  7. Optimization (Pareto)
  8. Uncertainty propagation
  9. All figures
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
from scipy.interpolate import griddata

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.validation import run_all_validation
from src.transport import run_transport_simulation
from src.thermal import run_thermal_simulation, ThermalResistanceNetwork, THERMAL_DEFAULTS
from src.thermoelectric import compute_teg_performance, TEG_DEFAULTS
from src.pressure_drop import compute_pressure_drop_full
from src.adsorption import DEFAULTS_HCHO, DEFAULTS_CH3CHO
from src.cycle_model import CycleConfig, run_single_evaluation, run_full_cycling
from src.sweep import run_sweep, generate_lhs_samples, sample_to_config
from src.sensitivity import run_sensitivity, sensitivity_to_dataframe, PROBLEM, OUTPUTS
from src.optimization import train_surrogate, find_pareto_front, run_uncertainty_propagation


COLORS = {
    'hcho': '#1a9988',       # teal
    'ch3cho': '#9b59b6',     # violet
    'teg': '#d4a017',        # gold
    'adsorption': '#2e86ab', # blue-green
    'regeneration': '#e07a5f',# warm orange
    'temperature': '#c44e52', # red
    'pressure': '#666666',   # gray
    'uncertainty': '#aaaaaa', # light gray
}

def setup_rcparams(dark=False):
    """Configure matplotlib for publication quality."""
    rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    if dark:
        rcParams.update({
            'figure.facecolor': '#0a1628',
            'axes.facecolor': '#0a1628',
            'text.color': '#e0e0e0',
            'axes.labelcolor': '#e0e0e0',
            'xtick.color': '#b0b0b0',
            'ytick.color': '#b0b0b0',
            'axes.edgecolor': '#444444',
        })
    else:
        rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'text.color': '#1a1a1a',
            'axes.labelcolor': '#1a1a1a',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'axes.edgecolor': '#333333',
        })


def save_figure(fig, name, formats=['png', 'svg', 'pdf'], dark=False):
    """Save figure in multiple formats."""
    base_dir = 'figures_dark' if dark else 'figures'
    os.makedirs(base_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(base_dir, f'{name}.{fmt}')
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"  Saved {base_dir}/{name} ({', '.join(formats)})")


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add panel label (a, b, c, ...) to axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='right')



def main():
    t_start_total = time.time()
    os.makedirs('data/generated/cycle_histories', exist_ok=True)
    os.makedirs('validation', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    # Phase 1: Validation
    print("\n" + "="*70)
    print("PHASE 1: VALIDATION")
    print("="*70)
    val = run_all_validation()

    # Save convergence tables
    val['grid_convergence'].to_csv('validation/grid_convergence.csv', index=False)
    val['timestep_convergence'].to_csv('validation/timestep_convergence.csv', index=False)
    val['teg_validation'].to_csv('validation/teg_validation.csv', index=False)
    val['pressure_drop'].to_csv('validation/pressure_drop.csv', index=False)
    print("Validation results saved.")

    # Phase 2: Baseline spatiotemporal results
    print("\n" + "="*70)
    print("PHASE 2: BASELINE SPATIOTEMPORAL SIMULATION")
    print("="*70)
    t_eval_baseline = np.linspace(0, 600, 200)
    baseline_result = run_transport_simulation(
        Nz=80, L=0.05, u=0.05, D_ax=1e-5, eps=0.40, rho_sorbent=600.0,
        params_hcho=DEFAULTS_HCHO.copy(), params_ch3cho=DEFAULTS_CH3CHO.copy(),
        C_in_hcho=0.04, C_in_ch3cho=0.02,
        t_span=(0, 600), t_eval=t_eval_baseline,
    )
    print(f"Baseline: solver_success={baseline_result['solver_success']}, nfev={baseline_result['nfev']}")
    print(f"  Final HCHO capture eff: {baseline_result['capture_eff_hcho'][-1]:.3f}")
    print(f"  Final CH3CHO capture eff: {baseline_result['capture_eff_ch3cho'][-1]:.3f}")

    # Thermal baseline
    thermal_baseline = run_thermal_simulation(
        Nz=80, L=0.05, T0=298.15,
        t_span=(0, 360), t_eval=np.linspace(0, 360, 150),
        Q_heater=2.0, phase='regeneration',
    )
    print(f"Thermal baseline: T_max = {np.max(thermal_baseline['T']):.1f} K")

    # Phase 3: Parameter sweep
    print("\n" + "="*70)
    print("PHASE 3: PARAMETER SWEEP (LHS)")
    print("="*70)
    sweep_df = run_sweep(n_samples=1500, n_jobs=1, seed=42)
    print(f"Sweep shape: {sweep_df.shape}")

    # Phase 4: Sensitivity analysis
    print("\n" + "="*70)
    print("PHASE 4: SOBOL SENSITIVITY ANALYSIS")
    print("="*70)
    si_results, si_raw, si_params = run_sensitivity(N=64, seed=42)
    si_df = sensitivity_to_dataframe(si_results)
    si_df.to_csv('data/generated/sobol_indices.csv', index=False)
    print("Sobol indices saved.")

    # Phase 5: Cyclic simulations
    print("\n" + "="*70)
    print("PHASE 5: CYCLIC SIMULATIONS")
    print("="*70)
    cycle_configs = {
        'baseline': dict(),
        'high_capture': dict(u=0.02, L=0.08, t_adsorption=900),
        'low_pressure_drop': dict(u=0.02, d_p=3.5e-3, eps=0.45),
        'high_teg': dict(T_exhaust=623.15, h_cold=40.0),
        'optimized': dict(u=0.03, L=0.06, T_exhaust=598.15, d_p=2.5e-3,
                         t_adsorption=720, t_desorption=360, h_cold=35.0),
    }
    cycle_results = {}
    for name, kwargs in cycle_configs.items():
        print(f"  Running {name} cycling (10 cycles)...")
        config = CycleConfig(**kwargs)
        cr = run_full_cycling(config, n_cycles=10)
        cycle_results[name] = cr
        # Save time history
        cycle_df = pd.DataFrame({
            't': cr['t'], 'C_out_hcho': cr['C_out_hcho'],
            'C_out_ch3cho': cr['C_out_ch3cho'],
            'q_mean_hcho': cr['q_mean_hcho'], 'q_mean_ch3cho': cr['q_mean_ch3cho'],
            'T_mean': cr['T_mean'], 'P_teg': cr['P_teg'],
        })
        cycle_df.to_csv(f'data/generated/cycle_histories/{name}.csv', index=False)
        cm = cr['cycle_metrics'][-1]
        print(f"    Cycle 10: cap_hcho={cm['capture_eff_hcho']:.3f}, "
              f"regen_hcho={cm['regen_completeness_hcho']:.3f}, "
              f"T_max={cm['T_max_regen']:.1f}K")

    # Phase 6: Surrogate model
    print("\n" + "="*70)
    print("PHASE 6: SURROGATE MODEL")
    print("="*70)
    input_cols = [c for c in sweep_df.columns if c.startswith('input_')]
    output_cols = ['capture_eff_hcho', 'capture_eff_ch3cho', 'dP', 'P_TEG', 'regen_frac_hcho']
    # Filter valid
    valid_mask = sweep_df['solver_success'] == True
    for c in output_cols:
        if c in sweep_df.columns:
            valid_mask &= sweep_df[c].notna()
    valid_cols = [c for c in output_cols if c in sweep_df.columns]
    valid_inputs = [c for c in input_cols if c in sweep_df.columns]
    
    surrogates = None
    if len(valid_inputs) > 0 and len(valid_cols) > 0:
        surrogates = train_surrogate(sweep_df[valid_mask], valid_inputs, valid_cols)

    # Phase 7: Pareto optimization
    print("\n" + "="*70)
    print("PHASE 7: MULTI-OBJECTIVE OPTIMIZATION")
    print("="*70)
    pareto_cols_max = ['capture_eff_hcho', 'regen_frac_hcho']
    pareto_cols_min = ['dP']
    pareto_cols_exist = [c for c in pareto_cols_max + pareto_cols_min if c in sweep_df.columns]
    pareto_df = None
    if len(pareto_cols_exist) == len(pareto_cols_max) + len(pareto_cols_min):
        pareto_df = find_pareto_front(sweep_df[valid_mask], pareto_cols_max, pareto_cols_min)
        pareto_df.to_csv('data/generated/pareto_front.csv', index=False)
        print(f"Pareto front: {len(pareto_df)} designs")

    # Phase 8: Uncertainty propagation
    print("\n" + "="*70)
    print("PHASE 8: UNCERTAINTY PROPAGATION")
    print("="*70)
    unc_df = run_uncertainty_propagation(dict(), n_mc=300, seed=42)
    unc_df.to_csv('data/generated/uncertainty_results.csv', index=False)
    valid_unc = unc_df[unc_df['solver_success'] == True]
    if len(valid_unc) > 0:
        for col in ['capture_eff_hcho', 'P_TEG', 'regen_frac_hcho', 'dP']:
            if col in valid_unc.columns:
                vals = valid_unc[col].dropna()
                if len(vals) > 0:
                    print(f"  {col}: median={vals.median():.4f}, "
                          f"5th={vals.quantile(0.05):.4f}, 95th={vals.quantile(0.95):.4f}")

    # Phase 9: Generate figures
    print("\n" + "="*70)
    print("PHASE 9: GENERATING FIGURES")
    print("="*70)

    for dark in [False, True]:
        setup_rcparams(dark=dark)
        suffix = '_dark' if dark else ''
        
        print(f"  Figure 2{suffix}: Validation...")
        fig2, axes2 = plt.subplots(2, 2, figsize=(7.5, 6))
        
        gc = val['grid_convergence']
        axes2[0,0].plot(gc['Nz'], gc['capture_eff_hcho'], 'o-', color=COLORS['hcho'], label='HCHO')
        axes2[0,0].plot(gc['Nz'], gc['capture_eff_ch3cho'], 's-', color=COLORS['ch3cho'], label='CH₃CHO')
        axes2[0,0].set_xlabel('Grid points $N_z$')
        axes2[0,0].set_ylabel('Mean capture efficiency')
        axes2[0,0].legend(frameon=False)
        add_panel_label(axes2[0,0], 'a')

        tc = val['timestep_convergence']
        axes2[0,1].semilogx(tc['rtol'], tc['capture_eff_hcho'], 'o-', color=COLORS['hcho'])
        axes2[0,1].set_xlabel('Solver tolerance (rtol)')
        axes2[0,1].set_ylabel('Mean capture efficiency')
        axes2[0,1].invert_xaxis()
        add_panel_label(axes2[0,1], 'b')

        teg_v = val['teg_validation']
        axes2[1,0].plot(teg_v['dT'], teg_v['P_max_W'], 'o-', color=COLORS['teg'])
        axes2[1,0].axhspan(0.5, 3.0, alpha=0.15, color=COLORS['teg'], label='Literature range (single module)')
        axes2[1,0].set_xlabel('$\\Delta T$ [K]')
        axes2[1,0].set_ylabel('$P_{max}$ [W]')
        axes2[1,0].legend(frameon=False, fontsize=6)
        add_panel_label(axes2[1,0], 'c')

        pd_v = val['pressure_drop']
        axes2[1,1].plot(pd_v['u'], pd_v['dP_Pa'], 'o-', color=COLORS['pressure'])
        axes2[1,1].set_xlabel('Superficial velocity [m/s]')
        axes2[1,1].set_ylabel('$\\Delta P$ [Pa]')
        add_panel_label(axes2[1,1], 'd')

        fig2.tight_layout()
        save_figure(fig2, f'figure_02_validation{suffix}', dark=dark)
        plt.close(fig2)

        print(f"  Figure 3{suffix}: Spatiotemporal adsorption...")
        fig3, axes3 = plt.subplots(2, 2, figsize=(7.5, 6))

        z = baseline_result['z'] * 100  # cm
        t = baseline_result['t'] / 60   # min
        C_h = baseline_result['C_hcho']
        im3a = axes3[0,0].pcolormesh(z, t, C_h, cmap='viridis', shading='auto')
        plt.colorbar(im3a, ax=axes3[0,0], label='$C_{HCHO}$ [mol/m³]')
        axes3[0,0].set_xlabel('Bed position [cm]')
        axes3[0,0].set_ylabel('Time [min]')
        axes3[0,0].set_title('Formaldehyde concentration', fontsize=8)
        add_panel_label(axes3[0,0], 'a')

        C_a = baseline_result['C_ch3cho']
        im3b = axes3[0,1].pcolormesh(z, t, C_a, cmap='magma', shading='auto')
        plt.colorbar(im3b, ax=axes3[0,1], label='$C_{CH_3CHO}$ [mol/m³]')
        axes3[0,1].set_xlabel('Bed position [cm]')
        axes3[0,1].set_ylabel('Time [min]')
        axes3[0,1].set_title('Acetaldehyde concentration', fontsize=8)
        add_panel_label(axes3[0,1], 'b')

        q_h = baseline_result['q_hcho']
        im3c = axes3[1,0].pcolormesh(z, t, q_h * 1e3, cmap='cividis', shading='auto')
        plt.colorbar(im3c, ax=axes3[1,0], label='$q_{HCHO}$ [mmol/g]')
        axes3[1,0].set_xlabel('Bed position [cm]')
        axes3[1,0].set_ylabel('Time [min]')
        axes3[1,0].set_title('HCHO sorbent loading', fontsize=8)
        add_panel_label(axes3[1,0], 'c')

        axes3[1,1].plot(t, baseline_result['breakthrough_hcho'], '-', color=COLORS['hcho'], label='HCHO', lw=1.5)
        axes3[1,1].plot(t, baseline_result['breakthrough_ch3cho'], '-', color=COLORS['ch3cho'], label='CH₃CHO', lw=1.5)
        axes3[1,1].axhline(0.05, color='gray', ls='--', lw=0.5, label='5% threshold')
        axes3[1,1].set_xlabel('Time [min]')
        axes3[1,1].set_ylabel('$C_{out}/C_{in}$')
        axes3[1,1].set_ylim(-0.02, 1.05)
        axes3[1,1].legend(frameon=False)
        add_panel_label(axes3[1,1], 'd')

        fig3.tight_layout()
        save_figure(fig3, f'figure_03_adsorption{suffix}', dark=dark)
        plt.close(fig3)

        print(f"  Figure 4{suffix}: Thermal + TEG...")
        fig4, axes4 = plt.subplots(2, 2, figsize=(7.5, 6))

        z_th = thermal_baseline['z'] * 100
        t_th = thermal_baseline['t'] / 60
        T_field = thermal_baseline['T'] - 273.15  # °C
        im4a = axes4[0,0].pcolormesh(z_th, t_th, T_field, cmap='inferno', shading='auto')
        plt.colorbar(im4a, ax=axes4[0,0], label='$T_{bed}$ [°C]')
        axes4[0,0].set_xlabel('Bed position [cm]')
        axes4[0,0].set_ylabel('Time [min]')
        add_panel_label(axes4[0,0], 'a')

        dT_range = np.linspace(20, 300, 100)
        P_teg_range = []
        for dT in dT_range:
            teg = compute_teg_performance(298.15 + dT, 298.15)
            P = float(np.atleast_1d(teg['P_max'])[0]) if np.ndim(teg['P_max']) > 0 else float(teg['P_max'])
            P_teg_range.append(P)
        axes4[0,1].plot(dT_range, P_teg_range, '-', color=COLORS['teg'], lw=1.5)
        axes4[0,1].fill_between(dT_range, 0, P_teg_range, alpha=0.15, color=COLORS['teg'])
        axes4[0,1].set_xlabel('$\\Delta T_{TEG}$ [K]')
        axes4[0,1].set_ylabel('$P_{max}$ [W]')
        axes4[0,1].set_title('TEG matched-load power', fontsize=8)
        add_panel_label(axes4[0,1], 'b')

        axes4[1,0].plot(t_th, thermal_baseline['T_mean'] - 273.15, '-', color=COLORS['temperature'], lw=1.5)
        axes4[1,0].axhline(25, color='gray', ls=':', lw=0.5, label='Ambient')
        axes4[1,0].set_xlabel('Time [min]')
        axes4[1,0].set_ylabel('Mean bed $T$ [°C]')
        axes4[1,0].legend(frameon=False)
        add_panel_label(axes4[1,0], 'c')

        T_range = np.linspace(300, 420, 100)
        k_des = DEFAULTS_HCHO['A_des'] * np.exp(-DEFAULTS_HCHO['E_a_des'] / (8.314 * T_range))
        regen_300s = 1 - np.exp(-k_des * 300)
        regen_600s = 1 - np.exp(-k_des * 600)
        axes4[1,1].plot(T_range - 273.15, regen_300s * 100, '-', color=COLORS['regeneration'], label='5 min', lw=1.5)
        axes4[1,1].plot(T_range - 273.15, regen_600s * 100, '--', color=COLORS['regeneration'], label='10 min', lw=1.5)
        axes4[1,1].set_xlabel('Regeneration temperature [°C]')
        axes4[1,1].set_ylabel('Regeneration completeness [%]')
        axes4[1,1].legend(frameon=False)
        add_panel_label(axes4[1,1], 'd')

        fig4.tight_layout()
        save_figure(fig4, f'figure_04_thermal_teg{suffix}', dark=dark)
        plt.close(fig4)

        print(f"  Figure 5{suffix}: Cyclic operation...")
        cr = cycle_results['baseline']
        fig5, axes5 = plt.subplots(4, 1, figsize=(7.5, 8), sharex=True)

        t_min = np.array(cr['t']) / 60
        phase_arr = np.array(cr['phase'])

        for ax in axes5:
            for i in range(len(phase_arr)-1):
                if phase_arr[i] == 'adsorption':
                    c = COLORS['adsorption']
                elif phase_arr[i] == 'regeneration':
                    c = COLORS['regeneration']
                else:
                    c = '#888888'
                ax.axvspan(t_min[i], t_min[min(i+1, len(t_min)-1)], alpha=0.05, color=c, lw=0)

        axes5[0].plot(t_min, cr['C_out_hcho'] / 0.04, '-', color=COLORS['hcho'], lw=0.8, label='HCHO')
        axes5[0].plot(t_min, cr['C_out_ch3cho'] / 0.02, '-', color=COLORS['ch3cho'], lw=0.8, label='CH₃CHO')
        axes5[0].set_ylabel('$C_{out}/C_{in}$')
        axes5[0].legend(frameon=False, loc='upper right')
        add_panel_label(axes5[0], 'a', x=-0.08)

        axes5[1].plot(t_min, cr['q_mean_hcho'] * 1e3, '-', color=COLORS['hcho'], lw=0.8)
        axes5[1].plot(t_min, cr['q_mean_ch3cho'] * 1e3, '-', color=COLORS['ch3cho'], lw=0.8)
        axes5[1].set_ylabel('$\\bar{q}$ [mmol/g]')
        add_panel_label(axes5[1], 'b', x=-0.08)

        axes5[2].plot(t_min, np.array(cr['T_mean']) - 273.15, '-', color=COLORS['temperature'], lw=0.8)
        axes5[2].set_ylabel('$\\bar{T}_{bed}$ [°C]')
        add_panel_label(axes5[2], 'c', x=-0.08)

        axes5[3].plot(t_min, cr['P_teg'], '-', color=COLORS['teg'], lw=0.8)
        axes5[3].set_ylabel('$P_{TEG}$ [W]')
        axes5[3].set_xlabel('Time [min]')
        add_panel_label(axes5[3], 'd', x=-0.08)

        fig5.tight_layout()
        save_figure(fig5, f'figure_05_cycling{suffix}', dark=dark)
        plt.close(fig5)

        print(f"  Figure 6{suffix}: Design space...")
        fig6, axes6 = plt.subplots(2, 2, figsize=(7.5, 6))
        
        valid_sweep = sweep_df[sweep_df['solver_success'] == True].copy()
        
        if len(valid_sweep) > 50:
            sc6a = axes6[0,0].scatter(valid_sweep['input_u'], valid_sweep['input_L']*100,
                                       c=valid_sweep['capture_eff_hcho'], cmap='viridis',
                                       s=8, alpha=0.6, vmin=0, vmax=1)
            plt.colorbar(sc6a, ax=axes6[0,0], label='HCHO capture eff.')
            axes6[0,0].set_xlabel('Velocity [m/s]')
            axes6[0,0].set_ylabel('Bed length [cm]')
            add_panel_label(axes6[0,0], 'a')

            sc6b = axes6[0,1].scatter(valid_sweep['input_u'], valid_sweep['input_d_p']*1000,
                                       c=np.log10(valid_sweep['dP'].clip(lower=0.1)),
                                       cmap='magma', s=8, alpha=0.6)
            plt.colorbar(sc6b, ax=axes6[0,1], label='log₁₀(ΔP [Pa])')
            axes6[0,1].set_xlabel('Velocity [m/s]')
            axes6[0,1].set_ylabel('Particle diameter [mm]')
            add_panel_label(axes6[0,1], 'b')

            sc6c = axes6[1,0].scatter(valid_sweep['input_T_exhaust']-273.15, valid_sweep['input_h_cold'],
                                       c=valid_sweep['P_TEG'], cmap='YlOrBr',
                                       s=8, alpha=0.6)
            plt.colorbar(sc6c, ax=axes6[1,0], label='$P_{TEG}$ [W]')
            axes6[1,0].set_xlabel('Exhaust temperature [°C]')
            axes6[1,0].set_ylabel('$h_{cold}$ [W/(m²·K)]')
            add_panel_label(axes6[1,0], 'c')

            sc6d = axes6[1,1].scatter(valid_sweep['input_t_desorption']/60,
                                       valid_sweep['input_T_exhaust']-273.15,
                                       c=valid_sweep['regen_frac_hcho'], cmap='RdYlGn',
                                       s=8, alpha=0.6, vmin=0, vmax=1)
            plt.colorbar(sc6d, ax=axes6[1,1], label='Regen. completeness')
            axes6[1,1].set_xlabel('Desorption duration [min]')
            axes6[1,1].set_ylabel('Exhaust temperature [°C]')
            add_panel_label(axes6[1,1], 'd')

        fig6.tight_layout()
        save_figure(fig6, f'figure_06_design_space{suffix}', dark=dark)
        plt.close(fig6)

        print(f"  Figure 7{suffix}: Sensitivity...")
        fig7, axes7 = plt.subplots(1, 2, figsize=(7.5, 4))

        if len(si_results) > 0:
            key = 'capture_eff_hcho' if 'capture_eff_hcho' in si_results else list(si_results.keys())[0]
            si = si_results[key]
            names = [n.replace('_', '\n') for n in si['names']]
            x_pos = np.arange(len(names))

            axes7[0].barh(x_pos, si['S1'], xerr=si['S1_conf'], height=0.6,
                          color=COLORS['adsorption'], alpha=0.8)
            axes7[0].set_yticks(x_pos)
            axes7[0].set_yticklabels(names, fontsize=6)
            axes7[0].set_xlabel('First-order Sobol index ($S_1$)')
            axes7[0].set_title(f'Sensitivity: {key}', fontsize=8)
            add_panel_label(axes7[0], 'a')

            axes7[1].barh(x_pos, si['ST'], xerr=si['ST_conf'], height=0.6,
                          color=COLORS['regeneration'], alpha=0.8)
            axes7[1].set_yticks(x_pos)
            axes7[1].set_yticklabels(names, fontsize=6)
            axes7[1].set_xlabel('Total-effect Sobol index ($S_T$)')
            axes7[1].set_title(f'Total effects: {key}', fontsize=8)
            add_panel_label(axes7[1], 'b')

        fig7.tight_layout()
        save_figure(fig7, f'figure_07_sensitivity{suffix}', dark=dark)
        plt.close(fig7)

        print(f"  Figure 8{suffix}: Optimization...")
        fig8, axes8 = plt.subplots(1, 2, figsize=(7.5, 3.5))

        if pareto_df is not None and len(pareto_df) > 0:
            sc8a = axes8[0].scatter(valid_sweep['dP'], valid_sweep['capture_eff_hcho'],
                                     c='#cccccc', s=4, alpha=0.3, label='All designs')
            if 'P_TEG' in pareto_df.columns:
                sc8p = axes8[0].scatter(pareto_df['dP'], pareto_df['capture_eff_hcho'],
                                         c=pareto_df['P_TEG'], cmap='YlOrBr', s=30,
                                         edgecolors='black', linewidths=0.5,
                                         zorder=5, label='Pareto optimal')
                plt.colorbar(sc8p, ax=axes8[0], label='$P_{TEG}$ [W]')
            axes8[0].set_xlabel('$\\Delta P$ [Pa]')
            axes8[0].set_ylabel('HCHO capture efficiency')
            axes8[0].legend(frameon=False, fontsize=6)
            add_panel_label(axes8[0], 'a')

            top_pareto = pareto_df.head(min(10, len(pareto_df)))
            par_cols = ['capture_eff_hcho', 'regen_frac_hcho', 'dP', 'P_TEG']
            par_cols = [c for c in par_cols if c in top_pareto.columns]
            if len(par_cols) >= 2:
                normed = top_pareto[par_cols].copy()
                for c in par_cols:
                    cmin, cmax = valid_sweep[c].min(), valid_sweep[c].max()
                    if cmax > cmin:
                        normed[c] = (normed[c] - cmin) / (cmax - cmin)
                    else:
                        normed[c] = 0.5
                for _, row in normed.iterrows():
                    axes8[1].plot(range(len(par_cols)), row[par_cols].values, '-o',
                                 color=COLORS['teg'], alpha=0.5, markersize=3)
                axes8[1].set_xticks(range(len(par_cols)))
                axes8[1].set_xticklabels([c.replace('_', '\n') for c in par_cols], fontsize=6)
                axes8[1].set_ylabel('Normalized value')
                axes8[1].set_title('Pareto-optimal designs', fontsize=8)
            add_panel_label(axes8[1], 'b')

        fig8.tight_layout()
        save_figure(fig8, f'figure_08_optimization{suffix}', dark=dark)
        plt.close(fig8)

        print(f"  Figure 9{suffix}: Uncertainty...")
        fig9, axes9 = plt.subplots(2, 2, figsize=(7.5, 6))

        if len(valid_unc) > 10:
            for ax_idx, (col, color, label) in enumerate([
                ('capture_eff_hcho', COLORS['hcho'], 'HCHO capture eff.'),
                ('P_TEG', COLORS['teg'], '$P_{TEG}$ [W]'),
                ('regen_frac_hcho', COLORS['regeneration'], 'Regen. completeness'),
                ('dP', COLORS['pressure'], '$\\Delta P$ [Pa]'),
            ]):
                ax = axes9[ax_idx // 2, ax_idx % 2]
                if col in valid_unc.columns:
                    vals = valid_unc[col].dropna()
                    if len(vals) > 5:
                        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
                        med = vals.median()
                        p5, p95 = vals.quantile(0.05), vals.quantile(0.95)
                        ax.axvline(med, color='black', ls='-', lw=1, label=f'Median: {med:.3f}')
                        ax.axvline(p5, color='black', ls='--', lw=0.5, label=f'5th: {p5:.3f}')
                        ax.axvline(p95, color='black', ls='--', lw=0.5, label=f'95th: {p95:.3f}')
                        ax.legend(frameon=False, fontsize=5)
                ax.set_xlabel(label)
                ax.set_ylabel('Count')
                add_panel_label(ax, chr(ord('a') + ax_idx))

        fig9.tight_layout()
        save_figure(fig9, f'figure_09_uncertainty{suffix}', dark=dark)
        plt.close(fig9)

        print(f"  Figure 1{suffix}: System overview...")
        fig1 = plt.figure(figsize=(7.5, 8))
        gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.35, wspace=0.3)

        ax1a = fig1.add_subplot(gs1[0, 0])
        ax1a.set_xlim(0, 10)
        ax1a.set_ylim(0, 10)
        ax1a.set_aspect('equal')
        ax1a.axis('off')
        txt_color = '#e0e0e0' if dark else '#1a1a1a'
        
        components = [
            (1, 7.5, 8, 1.5, 'Cookstove Exhaust', '#e07a5f'),
            (1, 5.5, 8, 1.5, 'TEG Module (Bi₂Te₃)', COLORS['teg']),
            (1, 3.5, 8, 1.5, 'Resistive Heater', COLORS['regeneration']),
            (1, 1.0, 8, 2.0, 'Amine-Functionalized\nPorous Sorbent Bed', COLORS['adsorption']),
        ]
        for x, y, w, h, label, color in components:
            rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor=txt_color, linewidth=0.8, alpha=0.6)
            ax1a.add_patch(rect)
            ax1a.text(x + w/2, y + h/2, label, ha='center', va='center',
                     fontsize=7, fontweight='bold', color=txt_color)
        
        for y_start, y_end in [(7.5, 7.0), (5.5, 5.0), (3.5, 3.0)]:
            ax1a.annotate('', xy=(5, y_end), xytext=(5, y_start),
                         arrowprops=dict(arrowstyle='->', color=txt_color, lw=1.2))
        
        ax1a.text(0.3, 2.0, 'Gas flow →', fontsize=6, color=txt_color, style='italic')
        add_panel_label(ax1a, 'a', x=0.0, y=1.02)

        ax1b = fig1.add_subplot(gs1[0, 1])
        ax1b.set_xlim(0, 10)
        ax1b.set_ylim(0, 10)
        ax1b.axis('off')
        ax1b.text(5, 9.5, 'Adsorption Mechanism', ha='center', fontsize=8, fontweight='bold', color=txt_color)
        ax1b.text(5, 8, 'R-NH₂ + HCHO → R-N=CH₂ + H₂O', ha='center', fontsize=8,
                 fontfamily='monospace', color=COLORS['hcho'])
        ax1b.text(5, 7, '(imine formation)', ha='center', fontsize=7, style='italic', color=txt_color)
        ax1b.text(5, 5.5, 'R-NH₂ + CH₃CHO → R-N=CHCH₃ + H₂O', ha='center', fontsize=8,
                 fontfamily='monospace', color=COLORS['ch3cho'])
        ax1b.text(5, 4.5, '(imine formation)', ha='center', fontsize=7, style='italic', color=txt_color)
        ax1b.text(5, 3, 'Regeneration (thermal):', ha='center', fontsize=7, fontweight='bold', color=txt_color)
        ax1b.text(5, 2, 'R-N=CH₂ + H₂O →$\\Delta$→ R-NH₂ + HCHO↑', ha='center', fontsize=7,
                 color=COLORS['regeneration'])
        add_panel_label(ax1b, 'b', x=0.0, y=1.02)

        ax1c = fig1.add_subplot(gs1[1, 0])
        ax1c.set_xlim(-0.5, 10.5)
        ax1c.set_ylim(-1, 5)
        ax1c.axis('off')
        ax1c.text(5, 4.5, 'Computational Domain', ha='center', fontsize=8, fontweight='bold', color=txt_color)
        bed = FancyBboxPatch((0.5, 1), 9, 2.5, boxstyle="round,pad=0.05",
                              facecolor=COLORS['adsorption'], alpha=0.2, edgecolor=txt_color, lw=0.8)
        ax1c.add_patch(bed)
        ax1c.annotate('', xy=(9.5, 2.25), xytext=(0.5, 2.25),
                      arrowprops=dict(arrowstyle='->', color=txt_color, lw=1.5))
        ax1c.text(5, 2.25, '$u \\cdot \\partial C/\\partial z$', ha='center', va='bottom',
                 fontsize=8, color=txt_color)
        ax1c.text(0.1, 2.25, '$C_{in}$', ha='right', fontsize=8, color=txt_color)
        ax1c.text(10, 2.25, '$C_{out}$', ha='left', fontsize=8, color=txt_color)
        ax1c.text(5, 0.5, '$z = 0$ to $z = L$', ha='center', fontsize=7, color=txt_color)
        ax1c.text(5, 3.7, '$D_{ax} \\partial^2 C/\\partial z^2 - R_{ads}$', ha='center',
                 fontsize=7, color=txt_color)
        add_panel_label(ax1c, 'c', x=-0.02, y=1.02)

        ax1d = fig1.add_subplot(gs1[1, 1])
        ax1d.set_xlim(0, 10)
        ax1d.set_ylim(0, 3)
        ax1d.axis('off')
        phases = [
            (0.2, 3.0, 'Adsorption', COLORS['adsorption']),
            (3.0, 1.5, 'Heating', COLORS['regeneration']),
            (4.5, 3.5, 'Desorption', COLORS['regeneration']),
            (8.0, 1.5, 'Cooling', '#888888'),
        ]
        for x_start, width, label, color in phases:
            rect = FancyBboxPatch((x_start, 0.8), width, 1.4, boxstyle="round,pad=0.05",
                                   facecolor=color, alpha=0.5, edgecolor=txt_color, lw=0.5)
            ax1d.add_patch(rect)
            ax1d.text(x_start + width/2, 1.5, label, ha='center', va='center',
                     fontsize=6, fontweight='bold', color=txt_color)
        for x in [3.0, 4.5, 8.0]:
            ax1d.annotate('', xy=(x+0.05, 1.5), xytext=(x-0.15, 1.5),
                         arrowprops=dict(arrowstyle='->', color=txt_color, lw=0.8))
        ax1d.text(5, 2.6, 'Cycle Timeline', ha='center', fontsize=8, fontweight='bold', color=txt_color)
        add_panel_label(ax1d, 'd', x=-0.02, y=1.02)

        fig1.tight_layout()
        save_figure(fig1, f'figure_01_overview{suffix}', dark=dark)
        plt.close(fig1)

    print("\n" + "="*70)
    print("SAVING LEDGERS")
    print("="*70)

    figure_entries = []
    for i in range(1, 10):
        name = f'figure_{i:02d}'
        figure_entries.append(dict(
            figure_id=f'Figure {i}', data_file='data/generated/',
            plot_script='run_all.py', result_type='MODEL OUTPUT',
            provenance='Generated from executed physics model'
        ))
    pd.DataFrame(figure_entries).to_csv('FIGURE_LEDGER.csv', index=False)

    results_entries = []
    bl_cr = cycle_results['baseline']['cycle_metrics'][-1]
    results_entries.extend([
        dict(claim='Baseline HCHO capture efficiency (cycle 10)', value=f"{bl_cr['capture_eff_hcho']:.3f}",
             units='-', source_script='run_all.py', output_file='data/generated/cycle_histories/baseline.csv',
             run_id='baseline_cycle10', conditions='Default parameters'),
        dict(claim='Baseline HCHO regeneration completeness', value=f"{bl_cr['regen_completeness_hcho']:.3f}",
             units='-', source_script='run_all.py', output_file='data/generated/cycle_histories/baseline.csv',
             run_id='baseline_cycle10', conditions='Default parameters'),
        dict(claim='Baseline max regeneration temperature', value=f"{bl_cr['T_max_regen']:.1f}",
             units='K', source_script='run_all.py', output_file='data/generated/cycle_histories/baseline.csv',
             run_id='baseline_cycle10', conditions='Default parameters'),
        dict(claim='Baseline TEG power', value=f"{bl_cr['P_TEG']:.2f}",
             units='W', source_script='run_all.py', output_file='data/generated/cycle_histories/baseline.csv',
             run_id='baseline_cycle10', conditions='Default parameters'),
        dict(claim='Total parameter sweep evaluations', value=str(len(sweep_df)),
             units='count', source_script='src/sweep.py', output_file='data/generated/full_parameter_sweep.csv',
             run_id='sweep_42', conditions='LHS, seed=42'),
        dict(claim='Successful sweep evaluations', value=str(len(sweep_df[sweep_df['solver_success']==True])),
             units='count', source_script='src/sweep.py', output_file='data/generated/full_parameter_sweep.csv',
             run_id='sweep_42', conditions='LHS, seed=42'),
    ])
    pd.DataFrame(results_entries).to_csv('RESULTS_LEDGER.csv', index=False)

    elapsed_total = time.time() - t_start_total
    print(f"\n{'='*70}")
    print(f"COMPLETE. Total time: {elapsed_total/60:.1f} minutes")
    print(f"{'='*70}")

    n_total_sims = len(sweep_df) + 300 + 5*10 + 64*(2*8+2)
    print(f"\nSimulations executed: ~{n_total_sims}")
    print(f"Figures generated: 9 × 2 (paper + dark) = 18 files")
    print(f"Sweep dataset: data/generated/full_parameter_sweep.csv")
    print(f"Source code: src/")


if __name__ == '__main__':
    main()
