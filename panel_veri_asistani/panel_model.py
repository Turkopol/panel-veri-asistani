# panel_model.py

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.tools.tools import add_constant
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan


def prepare_panel_data(df, id_col, time_col):
    """Veriyi panel veri formatına dönüştür."""
    df = df.copy()

    # Zaman sütununu numeric veya datetime yap
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            if df[time_col].isnull().all():
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        except:
            df[time_col] = pd.factorize(df[time_col])[0]

    df = df.set_index([id_col, time_col])
    return df


def run_fixed_effects(df, y_var, x_vars):
    """Sabit etkiler modeli (Fixed Effects)"""
    df_fe = df[[y_var] + x_vars].dropna()
    if df_fe.shape[0] <= len(x_vars) + 1:
        raise ValueError("Sabit etkiler modeli için yeterli gözlem yok.")
    model = PanelOLS(df_fe[y_var], add_constant(df_fe[x_vars]), entity_effects=True)
    results = model.fit(cov_type="robust")
    return results


def run_random_effects(df, y_var, x_vars):
    """Tesadüfi etkiler modeli (Random Effects)"""
    df_re = df[[y_var] + x_vars].dropna()
    if df_re.shape[0] <= len(x_vars) + 1:
        raise ValueError("Tesadüfi etkiler modeli için yeterli gözlem yok.")
    model = RandomEffects(df_re[y_var], add_constant(df_re[x_vars]))
    results = model.fit(cov_type="robust")
    return results


def hausman_test(fe_res, re_res):
    """Hausman testi: FE ve RE karşılaştırması"""
    b_fe = fe_res.params
    b_re = re_res.params

    common_coef = b_fe.index.intersection(b_re.index)
    b_diff = b_fe[common_coef] - b_re[common_coef]

    v_fe = fe_res.cov.loc[common_coef, common_coef]
    v_re = re_res.cov.loc[common_coef, common_coef]
    v_diff = v_fe - v_re

    try:
        inv_v_diff = np.linalg.inv(v_diff)
        stat = b_diff.T @ inv_v_diff @ b_diff
        df_h = len(b_diff)
        pval = 1 - stats.chi2.cdf(stat, df_h)
        return stat, pval
    except np.linalg.LinAlgError:
        return None, None


def breusch_pagan_test(df, y_var, x_vars):
    """Breusch-Pagan heteroskedastisite testi"""
    formula = f"{y_var} ~ {' + '.join(x_vars)}"
    model = ols(formula, data=df.dropna()).fit()
    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(model.resid, model.model.exog)
    return lm_stat, lm_pval, f_stat, f_pval


def wooldridge_test(panel_df, id_col, time_col, y_var, x_vars):
    """Wooldridge otokorelasyon testi (Basitleştirilmiş)"""
    try:
        df = panel_df.copy().reset_index()
        df = df.sort_values(by=[id_col, time_col])
        df["dy"] = df.groupby(id_col)[y_var].diff()
        for x in x_vars:
            df[f"d{x}"] = df.groupby(id_col)[x].diff()
        formula = f"dy ~ {' + '.join(['d' + x for x in x_vars])}"
        model = ols(formula, data=df.dropna()).fit()
        r_squared = model.rsquared
        n = df.dropna().shape[0]
        f_stat = r_squared * (n - len(x_vars) - 1) / (1 - r_squared)
        pval = 1 - stats.f.cdf(f_stat, len(x_vars), n - len(x_vars) - 1)
        return f_stat, pval
    except Exception:
        return None, None


