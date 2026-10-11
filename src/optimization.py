"""
Multi-objective optimization and surrogate model training.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_surrogate(df, input_cols, output_cols, seed=42):
    """Train gradient-boosted tree surrogates for each output."""
    df_valid = df.dropna(subset=output_cols + input_cols)
    if len(df_valid) < 50:
        print(f"Warning: only {len(df_valid)} valid samples")
        return None

    X = df_valid[input_cols].values
    surrogates = {}

    for col in output_cols:
        y = df_valid[col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed)

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=seed)
        model.fit(X_train2, y_train2)

        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        surrogates[col] = dict(
            model=model,
            r2_val=r2_score(y_val, y_pred_val),
            r2_test=r2_score(y_test, y_pred_test),
            mae_test=mean_absolute_error(y_test, y_pred_test),
            rmse_test=np.sqrt(mean_squared_error(y_test, y_pred_test)),
            feature_importance=dict(zip(input_cols, model.feature_importances_)),
            y_test=y_test, y_pred_test=y_pred_test,
        )
        print(f"  {col}: R²_test={surrogates[col]['r2_test']:.3f}, "
              f"MAE={surrogates[col]['mae_test']:.4f}, "
              f"RMSE={surrogates[col]['rmse_test']:.4f}")

    return surrogates


def find_pareto_front(df, obj_maximize, obj_minimize):
    """
    Extract Pareto-optimal designs from sweep data.

    Parameters
    ----------
    df : DataFrame
    obj_maximize : list of column names to maximize
    obj_minimize : list of column names to minimize
    """
    df_valid = df.dropna(subset=obj_maximize + obj_minimize).copy()

    obj_vals = np.zeros((len(df_valid), len(obj_maximize) + len(obj_minimize)))
    for i, col in enumerate(obj_maximize):
        obj_vals[:, i] = -df_valid[col].values  # negate for minimization
    for i, col in enumerate(obj_minimize):
        obj_vals[:, len(obj_maximize) + i] = df_valid[col].values

    is_pareto = np.ones(len(obj_vals), dtype=bool)
    for i in range(len(obj_vals)):
        if not is_pareto[i]:
            continue
        for j in range(len(obj_vals)):
            if i == j or not is_pareto[j]:
                continue
            if np.all(obj_vals[j] <= obj_vals[i]) and np.any(obj_vals[j] < obj_vals[i]):
                is_pareto[i] = False
                break

    pareto_df = df_valid.iloc[is_pareto].copy()
    pareto_df['pareto_rank'] = range(1, len(pareto_df) + 1)

    return pareto_df


def run_uncertainty_propagation(base_config_kwargs, n_mc=500, seed=42):
    """Monte Carlo uncertainty propagation on key literature parameters."""
    from src.cycle_model import CycleConfig, run_single_evaluation
    from src.adsorption import DEFAULTS_HCHO, DEFAULTS_CH3CHO

    np.random.seed(seed)

    uncertain = {
        'q_max_hcho': (1.4e-3, 0.3e-3),
        'K_L_hcho': (500.0, 150.0),
        'E_a_des_hcho': (65e3, 8e3),
        'k_LDF_hcho': (0.01, 0.003),
        'T_exhaust': (573.15, 50.0),
    }

    results = []
    for i in range(n_mc):
        kwargs = base_config_kwargs.copy()
        params_h = DEFAULTS_HCHO.copy()
        params_a = DEFAULTS_CH3CHO.copy()

        params_h['q_max'] = max(0.1e-3, np.random.normal(uncertain['q_max_hcho'][0], uncertain['q_max_hcho'][1]))
        params_h['K_L'] = max(50.0, np.random.normal(uncertain['K_L_hcho'][0], uncertain['K_L_hcho'][1]))
        params_h['E_a_des'] = max(30e3, np.random.normal(uncertain['E_a_des_hcho'][0], uncertain['E_a_des_hcho'][1]))
        params_h['k_LDF'] = max(0.001, np.random.normal(uncertain['k_LDF_hcho'][0], uncertain['k_LDF_hcho'][1]))
        kwargs['T_exhaust'] = max(423.15, np.random.normal(uncertain['T_exhaust'][0], uncertain['T_exhaust'][1]))

        # Scale CH3CHO relative to HCHO
        params_a['q_max'] = params_h['q_max'] * 0.71
        params_a['K_L'] = params_h['K_L'] * 0.70
        params_a['E_a_des'] = params_h['E_a_des'] * 0.89
        params_a['k_LDF'] = params_h['k_LDF'] * 0.80

        kwargs['params_hcho'] = params_h
        kwargs['params_ch3cho'] = params_a
        kwargs['Nz'] = 25
        kwargs['Nt_per_phase'] = 50

        config = CycleConfig(**kwargs)
        r = run_single_evaluation(config)
        r['mc_sample'] = i
        results.append(r)

    return pd.DataFrame(results)
