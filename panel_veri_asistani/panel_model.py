# panel_model.py

import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.tools.tools import add_constant
from scipy import stats


def prepare_panel_data(df, id_col, time_col):
    """Veriyi panel veri formatına dönüştür."""
    df = df.copy()
    df = df.set_index([id_col, time_col])
    return df


def run_fixed_effects(df, y_var, x_vars):
    """Sabit etkiler modeli (Fixed Effects)"""
    df_fe = df[[y_var] + x_vars].dropna()
    model = PanelOLS(df_fe[y_var], add_constant(df_fe[x_vars]), entity_effects=True)
    results = model.fit(cov_type="robust")
    return results


def run_random_effects(df, y_var, x_vars):
    """Tesadüfi etkiler modeli (Random Effects)"""
    df_re = df[[y_var] + x_vars].dropna()
    model = RandomEffects(df_re[y_var], add_constant(df_re[x_vars]))
    results = model.fit(cov_type="robust")
    return results


def hausman_test(fe_res, re_res):
    """Hausman testi: FE ve RE modellerini karşılaştırır."""
    b_fe = fe_res.params
    b_re = re_res.params

    # Ortak katsayılar
    common_coef = b_fe.index.intersection(b_re.index)
    b_diff = b_fe[common_coef] - b_re[common_coef]

    # Varyans-fark matrisi
    v_fe = fe_res.cov
    v_re = re_res.cov
    v_diff = v_fe.loc[common_coef, common_coef] - v_re.loc[common_coef, common_coef]

    # Hausman testi istatistiği
    try:
        stat = b_diff.T @ pd.linalg.inv(v_diff) @ b_diff
        df_h = len(b_diff)
        pval = 1 - stats.chi2.cdf(stat, df_h)
        return stat, pval
    except Exception as e:
        return None, None
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols

def breusch_pagan_test(df, y_var, x_vars):
    """Breusch-Pagan heteroskedastisite testi"""
    formula = f"{y_var} ~ {' + '.join(x_vars)}"
    model = ols(formula, data=df).fit()
    lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(model.resid, model.model.exog)
    return lm_stat, lm_pval, f_stat, f_pval


def wooldridge_test(panel_df, id_col, time_col, y_var, x_vars):
    """Wooldridge otokorelasyon testi (Basitleştirilmiş versiyon)"""
    # Kaynak: Drukker (2003) yöntemi
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
    except Exception as e:
        return None, None

