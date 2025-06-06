# grafikler.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

def plot_scatter(df, x, y):
    """Bağımlı vs. bağımsız değişken scatter plot"""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"{y} vs. {x}")
    return fig


def plot_trend(df, id_col, time_col, y):
    """Panel birimlerine göre zaman serisi trend çizgileri"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, grp in df.groupby(id_col):
        grp_sorted = grp.sort_values(by=time_col)
        ax.plot(grp_sorted[time_col], grp_sorted[y], label=str(key), alpha=0.5)
    ax.set_title(f"Zamana Göre {y} Trendleri (Panel Birimlerine Göre)")
    ax.set_xlabel("Zaman")
    ax.set_ylabel(y)
    ax.legend(loc="best", fontsize="small", frameon=False)
    return fig


def plot_residuals(y_true, y_pred):
    """Tahmin - Gerçek = Artıklar için grafik"""
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.residplot(x=y_pred, y=residuals, lowess=True, ax=ax)
    ax.set_title("Artık (Residual) Grafiği")
    ax.set_xlabel("Tahmin Edilen Değerler")
    ax.set_ylabel("Artıklar")
    return fig


def plot_residual_histogram(y_true, y_pred):
    """Artıkların histogramı"""
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=20, kde=True, ax=ax)
    ax.set_title("Artıkların Dağılımı (Histogram)")
    ax.set_xlabel("Artık")
    return fig
