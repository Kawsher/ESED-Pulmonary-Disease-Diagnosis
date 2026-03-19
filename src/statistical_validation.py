# =============================================================
# statistical_validation.py — Six Statistical Tests
# =============================================================
# Tests:
#   1. McNemar's test     — pairwise error pattern comparison
#   2. Friedman's test    — overall classifier ranking
#   3. Nemenyi post-hoc  — pairwise post-Friedman
#   4. Wilcoxon test     — pairwise CV score comparison
#   5. DeLong's test     — AUC comparison
#   6. Bootstrap CI      — 95% confidence intervals
# =============================================================

import numpy as np
import pandas as pd
import json
import os
from scipy import stats
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import mcnemar
from scikit_posthocs import posthoc_nemenyi_friedman

from config import CLASSES, NUM_CLASSES, SEED, METRICS_DIR


# ── Test 1: McNemar's Test ────────────────────────────────────

def mcnemar_test(
        y_true        : np.ndarray,
        y_pred_ensemble: np.ndarray,
        y_pred_base   : np.ndarray,
        model_name    : str
) -> dict:
    """
    McNemar's test comparing ensemble vs one base model.

    H0: No significant difference in error patterns.
    H1: Significant difference exists.

    Returns:
        Dict with chi2, p_value, significant
    """
    correct_ens  = (y_pred_ensemble == y_true)
    correct_base = (y_pred_base     == y_true)

    # Contingency table
    # b = base wrong, ensemble right
    # c = base right, ensemble wrong
    b = np.sum( correct_ens & ~correct_base)
    c = np.sum(~correct_ens &  correct_base)

    # With continuity correction
    if (b + c) == 0:
        chi2, p = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p    = stats.chi2.sf(chi2, df=1)

    return {
        'comparison' : f'ESED vs {model_name}',
        'b'          : int(b),
        'c'          : int(c),
        'chi2'       : round(chi2, 4),
        'p_value'    : round(p, 6),
        'significant': bool(p < 0.05),
    }


def run_mcnemar_all(
        y_true         : np.ndarray,
        y_pred_ensemble: np.ndarray,
        base_predictions: dict,
        save           : bool = True
) -> pd.DataFrame:
    """
    Run McNemar's test for all 5 base model comparisons.

    Args:
        y_true           : True labels
        y_pred_ensemble  : Ensemble predictions
        base_predictions : Dict {model_name: y_pred}

    Returns:
        DataFrame with test results
    """
    rows = []
    print("\nMcNemar's Test Results:")
    print("=" * 55)
    for model_name, y_pred in base_predictions.items():
        result = mcnemar_test(
            y_true, y_pred_ensemble, y_pred, model_name)
        rows.append(result)
        sig = '✅' if result['significant'] else '❌'
        print(f"  {result['comparison']:<28}: "
              f"χ²={result['chi2']:>7.3f} "
              f"p={result['p_value']:.6f} {sig}")

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(
            METRICS_DIR+'test1_mcnemar.csv', index=False)
    return df


# ── Test 2: Friedman's Test ───────────────────────────────────

def friedman_test(
        cv_scores_matrix: np.ndarray,
        classifier_names: list[str],
        save            : bool = True
) -> dict:
    """
    Friedman's test across all classifier CV scores.

    Args:
        cv_scores_matrix: Shape (n_classifiers, n_folds)
        classifier_names: Names of classifiers

    Returns:
        Dict with statistic, p_value, significant
    """
    stat, p = stats.friedmanchisquare(
        *cv_scores_matrix)

    result = {
        'statistic'  : round(float(stat), 4),
        'p_value'    : round(float(p),    6),
        'significant': bool(p < 0.05),
        'df'         : len(classifier_names) - 1,
    }

    print(f"\nFriedman's Test:")
    print(f"  χ²({result['df']}) = {result['statistic']}")
    print(f"  p = {result['p_value']}")
    sig = '✅ Significant' if result['significant'] \
        else '❌ Not significant'
    print(f"  {sig}")

    if save:
        pd.DataFrame([result]).to_csv(
            METRICS_DIR+'test2_friedman.csv', index=False)

    return result


# ── Test 3: Nemenyi Post-Hoc ──────────────────────────────────

def nemenyi_test(
        cv_scores_matrix: np.ndarray,
        classifier_names: list[str],
        save            : bool = True
) -> pd.DataFrame:
    """
    Nemenyi post-hoc test following significant Friedman.

    Returns:
        DataFrame of pairwise p-values
    """
    # Input: (n_folds, n_classifiers)
    scores_T = cv_scores_matrix.T
    df_nem   = posthoc_nemenyi_friedman(scores_T)
    df_nem.columns = classifier_names
    df_nem.index   = classifier_names

    print(f"\nNemenyi Post-Hoc Test (p-values):")
    print(f"  Significant pairs (p<0.05):")
    for i, c1 in enumerate(classifier_names):
        for j, c2 in enumerate(classifier_names):
            if i < j:
                p = df_nem.loc[c1, c2]
                if p < 0.05:
                    print(f"    {c1} vs {c2}: p={p:.4f} ✅")

    if save:
        df_nem.to_csv(
            METRICS_DIR+'test3_nemenyi.csv')

    return df_nem


# ── Test 4: Wilcoxon Signed-Rank Test ────────────────────────

def wilcoxon_test(
        scores_best  : np.ndarray,
        scores_others: dict,
        best_name    : str,
        save         : bool = True
) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test comparing best classifier
    vs all others on fold-level CV scores.

    Returns:
        DataFrame with test results
    """
    rows = []
    print(f"\nWilcoxon Test ({best_name} vs others):")

    for name, scores in scores_others.items():
        if name == best_name:
            continue
        try:
            stat, p = stats.wilcoxon(
                scores_best, scores)
            sig = '✅' if p < 0.05 else '❌'
            print(f"  vs {name:<22}: "
                  f"stat={stat:.3f} p={p:.4f} {sig}")
            rows.append({
                'Comparison' : f'{best_name} vs {name}',
                'Statistic'  : round(stat, 4),
                'p_value'    : round(p,    4),
                'Significant': bool(p < 0.05),
            })
        except Exception as e:
            print(f"  vs {name}: FAILED ({str(e)[:30]})")

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(
            METRICS_DIR+'test4_wilcoxon.csv', index=False)
    return df


# ── Test 5: DeLong's Test ─────────────────────────────────────

def delong_test(
        y_true         : np.ndarray,
        y_prob_ensemble: np.ndarray,
        base_probs     : dict,
        save           : bool = True
) -> pd.DataFrame:
    """
    DeLong's test comparing AUC between ensemble
    and each base model using variance estimation.

    Returns:
        DataFrame with AUC comparison results
    """
    def compute_auc_var(y_bin, y_prob):
        """Compute AUC and structural components."""
        n_pos    = y_bin.sum()
        n_neg    = len(y_bin) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5, 0
        auc_val  = roc_auc_score(y_bin, y_prob)
        # Hanley-McNeil variance approximation
        q1       = auc_val / (2 - auc_val)
        q2       = 2 * auc_val**2 / (1 + auc_val)
        var      = (auc_val*(1-auc_val) +
                    (n_pos-1)*(q1-auc_val**2) +
                    (n_neg-1)*(q2-auc_val**2)) / \
                   (n_pos * n_neg)
        return auc_val, var

    y_bin_all = label_binarize(
        y_true, classes=range(NUM_CLASSES))

    rows = []
    print("\nDeLong's AUC Test:")

    for model_name, y_prob_base in base_probs.items():
        auc_ens_vals = []
        auc_bas_vals = []
        z_vals       = []

        for ci in range(NUM_CLASSES):
            auc_e, var_e = compute_auc_var(
                y_bin_all[:, ci],
                y_prob_ensemble[:, ci])
            auc_b, var_b = compute_auc_var(
                y_bin_all[:, ci],
                y_prob_base[:, ci])

            auc_ens_vals.append(auc_e)
            auc_bas_vals.append(auc_b)

            if var_e + var_b > 0:
                z = (auc_e-auc_b) / \
                    np.sqrt(var_e + var_b)
                z_vals.append(z)

        mean_z = np.mean(z_vals) if z_vals else 0
        p_val  = 2 * (1 - stats.norm.cdf(abs(mean_z)))
        sig    = '✅' if p_val < 0.05 else '❌'

        print(f"  vs {model_name:<16}: "
              f"z={mean_z:.3f} p={p_val:.4f} {sig}")

        rows.append({
            'Comparison'  : f'ESED vs {model_name}',
            'AUC_Ensemble': round(
                np.mean(auc_ens_vals), 4),
            'AUC_Base'    : round(
                np.mean(auc_bas_vals), 4),
            'z_stat'      : round(mean_z, 4),
            'p_value'     : round(p_val,  4),
            'Significant' : bool(p_val < 0.05),
        })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(
            METRICS_DIR+'test5_delong.csv', index=False)
    return df


# ── Test 6: Bootstrap Confidence Intervals ────────────────────

def bootstrap_ci(
        y_true     : np.ndarray,
        y_pred     : np.ndarray,
        n_bootstrap: int = 1000,
        alpha      : float = 0.95,
        save       : bool = True,
        model_name : str = 'ESED_Ensemble'
) -> dict:
    """
    Compute bootstrap 95% CI for macro F1-score.

    Args:
        y_true     : True labels
        y_pred     : Predicted labels
        n_bootstrap: Number of bootstrap resamples

    Returns:
        Dict with mean, ci_lower, ci_upper
    """
    rng     = np.random.default_rng(SEED)
    f1_boot = []

    for _ in range(n_bootstrap):
        idx = rng.integers(
            0, len(y_true), len(y_true))
        f1  = f1_score(
            y_true[idx], y_pred[idx],
            average='macro', zero_division=0)
        f1_boot.append(f1)

    f1_boot  = np.array(f1_boot)
    lower_p  = (1 - alpha) / 2 * 100
    upper_p  = (1 + alpha) / 2 * 100
    ci_lower = np.percentile(f1_boot, lower_p)
    ci_upper = np.percentile(f1_boot, upper_p)
    mean_f1  = f1_boot.mean()

    result = {
        'model'    : model_name,
        'mean_f1'  : round(mean_f1,  4),
        'ci_lower' : round(ci_lower, 4),
        'ci_upper' : round(ci_upper, 4),
        'ci_width' : round(ci_upper - ci_lower, 4),
        'alpha'    : alpha,
        'n_boot'   : n_bootstrap,
    }

    print(f"\nBootstrap CI ({model_name}):")
    print(f"  Mean F1  : {result['mean_f1']}")
    print(f"  95% CI   : [{result['ci_lower']}, "
          f"{result['ci_upper']}]")

    if save:
        pd.DataFrame([result]).to_csv(
            METRICS_DIR+'test6_bootstrap_ci.csv',
            index=False)

    return result


def bootstrap_ci_all_models(
        y_true         : np.ndarray,
        y_pred_ensemble: np.ndarray,
        base_predictions: dict,
        n_bootstrap    : int = 1000
) -> pd.DataFrame:
    """
    Compute bootstrap CIs for ensemble and all base models.

    Returns:
        DataFrame with CI for all models
    """
    rows = []
    all_preds = {'ESED_Ensemble': y_pred_ensemble}
    all_preds.update(base_predictions)

    for name, y_pred in all_preds.items():
        result = bootstrap_ci(
            y_true, y_pred,
            n_bootstrap=n_bootstrap,
            model_name=name,
            save=False)
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(
        METRICS_DIR+'test6_bootstrap_ci.csv',
        index=False)

    print("\nBootstrap CI Summary:")
    print(f"  {'Model':<20} {'Mean F1':>8} "
          f"{'CI Lower':>10} {'CI Upper':>10}")
    print("  " + "-"*50)
    for _, row in df.iterrows():
        print(f"  {row['model']:<20} "
              f"{row['mean_f1']:>8.4f} "
              f"{row['ci_lower']:>10.4f} "
              f"{row['ci_upper']:>10.4f}")

    return df


if __name__ == '__main__':
    print("Statistical validation module loaded.")
    print("Tests: McNemar, Friedman, Nemenyi, "
          "Wilcoxon, DeLong, Bootstrap CI")
