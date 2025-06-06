# app.py

import streamlit as st
import pandas as pd
import statsmodels.api as sm
from grafikler import plot_scatter, plot_trend, plot_residuals, plot_residual_histogram
from panel_model import prepare_panel_data, run_fixed_effects, run_random_effects, hausman_test
from utils import create_excel_report
from panel_model import breusch_pagan_test, wooldridge_test


st.set_page_config(page_title="Panel Veri Analizi Asistanı", layout="wide")
st.title("📊 Panel Veri Analizi Asistanı")

st.markdown("""
Bu uygulama, panel veri setinizi yükleyerek sabit etkiler, tesadüfi etkiler gibi regresyon analizlerini gerçekleştirmenize olanak tanır. Lütfen veri dosyanızı yükleyin ve değişkenleri seçin.
""")

# Veri Yükleme
uploaded_file = st.file_uploader("📁 Veri dosyanızı yükleyin (CSV veya Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Dosya uzantısına göre oku
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Veri başarıyla yüklendi!")
    st.write("🔍 İlk 5 satır:")
    st.dataframe(df.head())

    st.markdown("### 🔧 Değişken Seçimi")
    columns = df.columns.tolist()

    id_col = st.selectbox("📌 Panel birimi (örneğin: banka, ülke):", columns)
    time_col = st.selectbox("⏳ Zaman değişkeni (örneğin: yıl):", [col for col in columns if col != id_col])
    y_var = st.selectbox("🎯 Bağımlı değişken (Y):", [col for col in columns if col not in [id_col, time_col]])
    x_vars = st.multiselect("📈 Bağımsız değişkenler (X):", [col for col in columns if col not in [y_var, id_col, time_col]])

    if st.button("✅ Analize Başla"):
        if not x_vars:
            st.warning("Lütfen en az bir bağımsız değişken seçin.")
        else:
            st.session_state["df"] = df
            st.session_state["id_col"] = id_col
            st.session_state["time_col"] = time_col
            st.session_state["y_var"] = y_var
            st.session_state["x_vars"] = x_vars
            st.success("Hazır! Şimdi analiz ve modelleme adımına geçebiliriz.")

else:
    st.info("Lütfen analiz için bir veri dosyası yükleyin.")

# Eğer tüm seçimler yapılmışsa analiz başlasın
if all(key in st.session_state for key in ["df", "id_col", "time_col", "y_var", "x_vars"]):
    st.markdown("## 📊 Regresyon Sonuçları")

    df = st.session_state["df"]
    id_col = st.session_state["id_col"]
    time_col = st.session_state["time_col"]
    y_var = st.session_state["y_var"]
    x_vars = st.session_state["x_vars"]

    try:
        # Panel veri formatına çevir
        panel_df = prepare_panel_data(df, id_col, time_col)

        # Sabit etkiler modeli
        fe_res = run_fixed_effects(panel_df, y_var, x_vars)
        st.subheader("📌 Sabit Etkiler Modeli (Fixed Effects)")
        st.text(str(fe_res.summary()))

        # Tesadüfi etkiler modeli
        re_res = run_random_effects(panel_df, y_var, x_vars)
        st.subheader("📌 Tesadüfi Etkiler Modeli (Random Effects)")
        st.text(str(re_res.summary()))

        # Hausman testi
        st.subheader("🏁 Hausman Testi")
        stat, pval = hausman_test(fe_res, re_res)
        if stat is not None:
            st.write(f"Hausman testi istatistiği: {stat:.4f}")
            st.write(f"p-değeri: {pval:.4f}")
            if pval < 0.05:
                st.success("Sabit etkiler modeli tercih edilmelidir. (p < 0.05)")
            else:
                st.info("Tesadüfi etkiler modeli tercih edilebilir. (p ≥ 0.05)")
        else:
            st.warning("Hausman testi hesaplanırken bir hata oluştu. Varyans farkı singular olabilir.")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

    st.markdown("---")
    st.markdown("## 📈 Grafikler")

    # Trend çizgisi
    st.subheader("🕒 Zaman Serisi Trend Grafiği")
    fig_trend = plot_trend(df, id_col, time_col, y_var)
    st.pyplot(fig_trend)

    # Scatter plotlar (Y vs X)
    st.subheader("📌 Bağımlı vs. Bağımsız Değişkenler (Scatter Plot)")
    for x in x_vars:
        st.markdown(f"**{y_var} vs. {x}**")
        fig_scatter = plot_scatter(df, x, y_var)
        st.pyplot(fig_scatter)

    # Artıklar (residuals)
    st.subheader("📉 Artık Analizi")
    y_true = fe_res.model.dependent.data
    y_pred = fe_res.predict().fitted_values

    fig_resid = plot_residuals(y_true, y_pred)
    st.pyplot(fig_resid)

    fig_hist = plot_residual_histogram(y_true, y_pred)
    st.pyplot(fig_hist)

    st.markdown("---")
    st.markdown("## 🔎 Ekonometrik Testler")

    # Breusch-Pagan testi
    st.subheader("📌 Heteroskedastisite Testi (Breusch-Pagan)")
    lm_stat, lm_pval, f_stat, f_pval = breusch_pagan_test(df, y_var, x_vars)
    st.write(f"LM istatistiği: {lm_stat:.4f}, p-değeri: {lm_pval:.4f}")
    if lm_pval < 0.05:
        st.warning("Heteroskedastisite mevcuttur (p < 0.05).")
    else:
        st.success("Heteroskedastisite bulunmamaktadır (p ≥ 0.05).")

    # Wooldridge testi
    st.subheader("📌 Otokorelasyon Testi (Wooldridge)")
    wool_stat, wool_pval = wooldridge_test(panel_df, id_col, time_col, y_var, x_vars)
    if wool_stat is not None:
        st.write(f"F istatistiği: {wool_stat:.4f}, p-değeri: {wool_pval:.4f}")
        if wool_pval < 0.05:
            st.warning("1. dereceden otokorelasyon mevcuttur (p < 0.05).")
        else:
            st.success("Otokorelasyon bulunmamaktadır (p ≥ 0.05).")
    else:
        st.error("Wooldridge testi hesaplanırken hata oluştu.")

    st.markdown("---")
    st.markdown("## 📤 Sonuçları Dışa Aktar")

    try:
        excel_output = create_excel_report(
            fe_res, re_res,
            (stat, pval),
            (lm_stat, lm_pval),
            (wool_stat, wool_pval)
        )
        st.download_button(
            label="📥 Excel Olarak İndir (.xlsx)",
            data=excel_output,
            file_name="panel_veri_analizi_sonuclari.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Dışa aktarım sırasında hata oluştu: {e}")


