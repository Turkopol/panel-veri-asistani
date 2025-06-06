# app.py

import streamlit as st
import pandas as pd

from panel_model import (
    prepare_panel_data,
    run_fixed_effects,
    run_random_effects,
    hausman_test,
    breusch_pagan_test,
    wooldridge_test
)

from grafikler import (
    plot_scatter,
    plot_trend,
    plot_residuals,
    plot_residual_histogram
)

from utils import create_excel_report

st.set_page_config(page_title="Panel Veri Analizi Asistanı", layout="wide")
st.title("📊 Panel Veri Analizi Asistanı")

st.markdown("""
Bu uygulama, panel veri setinizi yükleyerek sabit ve tesadüfi etkiler modelleri kurmanıza, Hausman testi ile model seçimi yapmanıza, ekonometrik testler gerçekleştirmenize ve grafiklerle sonuçları görselleştirmenize olanak tanır.
""")

uploaded_file = st.file_uploader("📁 Veri dosyanızı yükleyin (CSV veya Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Veri başarıyla yüklendi!")
    st.dataframe(df.head())

    st.markdown("### 🔧 Değişken Seçimi")
    columns = df.columns.tolist()

    id_col = st.selectbox("📌 Panel birimi (örnek: firma, ülke)", columns)
    time_col = st.selectbox("⏳ Zaman değişkeni (örnek: yıl)", columns)
    y_var = st.selectbox("🎯 Bağımlı değişken (Y)", columns)
    x_vars = st.multiselect("📈 Bağımsız değişkenler (X)", [col for col in columns if col not in [y_var, id_col, time_col]])

    # 🎯 Zaman sütununu dönüştür
    if time_col:
        if not pd.api.types.is_numeric_dtype(df[time_col]):
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                if df[time_col].isnull().all():
                    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            except:
                df[time_col] = pd.factorize(df[time_col])[0]

    # 📋 Tasviri İstatistikler
    if y_var and x_vars:
        st.markdown("## 📋 Tasviri İstatistikler")
        selected_cols = [y_var] + x_vars
        try:
            desc_stats = df[selected_cols].describe().T
            desc_stats = desc_stats.rename(columns={
                "count": "Gözlem Sayısı",
                "mean": "Ortalama",
                "std": "Std. Sapma",
                "min": "Min",
                "25%": "1. Çeyrek",
                "50%": "Medyan",
                "75%": "3. Çeyrek",
                "max": "Maksimum"
            })
            st.dataframe(desc_stats.style.format("{:.3f}"))
        except Exception as e:
            st.warning(f"Tasviri istatistikler hesaplanamadı: {e}")

    if st.button("✅ Analizi Başlat"):
        if not x_vars:
            st.warning("Lütfen en az bir bağımsız değişken seçin.")
        else:
            try:
                panel_df = prepare_panel_data(df, id_col, time_col)

                st.markdown("## 📊 Regresyon Sonuçları")
                fe_res = run_fixed_effects(panel_df, y_var, x_vars)
                st.subheader("📌 Sabit Etkiler Modeli (Fixed Effects)")
                st.code(fe_res.summary.as_text())

                re_res = run_random_effects(panel_df, y_var, x_vars)
                st.subheader("📌 Tesadüfi Etkiler Modeli (Random Effects)")
                st.code(re_res.summary.as_text())

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
                    st.warning("Hausman testi hesaplanamadı. Varyans farkı matrisi terslenemiyor.")

                st.markdown("## 🔎 Ekonometrik Testler")
                st.subheader("📌 Heteroskedastisite Testi (Breusch-Pagan)")
                lm_stat, lm_pval, f_stat, f_pval = breusch_pagan_test(df, y_var, x_vars)
                st.write(f"LM istatistiği: {lm_stat:.4f}, p-değeri: {lm_pval:.4f}")
                if lm_pval < 0.05:
                    st.warning("Heteroskedastisite mevcuttur (p < 0.05).")
                else:
                    st.success("Heteroskedastisite bulunmamaktadır (p ≥ 0.05).")

                st.subheader("📌 Otokorelasyon Testi (Wooldridge)")
                wool_stat, wool_pval = wooldridge_test(panel_df, id_col, time_col, y_var, x_vars)
                if wool_stat is not None:
                    st.write(f"F istatistiği: {wool_stat:.4f}, p-değeri: {wool_pval:.4f}")
                    if wool_pval < 0.05:
                        st.warning("1. dereceden otokorelasyon mevcuttur (p < 0.05).")
                    else:
                        st.success("Otokorelasyon bulunmamaktadır (p ≥ 0.05).")
                else:
                    st.error("Wooldridge testi hesaplanamadı.")

                st.markdown("## 📈 Grafikler")
                st.subheader("🕒 Zaman Serisi Trend Grafiği")
                st.pyplot(plot_trend(df, id_col, time_col, y_var))

                st.subheader("📌 Bağımlı vs. Bağımsız Değişkenler (Scatter Plot)")
                for x in x_vars:
                    st.markdown(f"**{y_var} vs. {x}**")
                    st.pyplot(plot_scatter(df, x, y_var))

                st.subheader("📉 Artık Analizi")
                y_true = fe_res.model.dependent.data
                y_pred = fe_res.predict().fitted_values
                st.pyplot(plot_residuals(y_true, y_pred))
                st.pyplot(plot_residual_histogram(y_true, y_pred))

                st.markdown("## 📤 Sonuçları Dışa Aktar")
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
                st.error(f"Bir hata oluştu: {e}")



