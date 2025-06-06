# utils.py

import pandas as pd
import xlsxwriter
import io

def create_excel_report(fe_res, re_res, hausman_result, bp_result, wool_result):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Fixed Effects
        fe_summary_df = fe_res.summary.tables[1]
        fe_summary_df.to_excel(writer, sheet_name="Sabit Etkiler")

        # Random Effects
        re_summary_df = re_res.summary.tables[1]
        re_summary_df.to_excel(writer, sheet_name="Tesadüfi Etkiler")

        # Hausman testi
        hausman_df = pd.DataFrame({
            "Test İstatistiği": [hausman_result[0]],
            "p-değeri": [hausman_result[1]]
        })
        hausman_df.to_excel(writer, sheet_name="Hausman Testi", index=False)

        # Breusch-Pagan
        bp_df = pd.DataFrame({
            "LM İstatistiği": [bp_result[0]],
            "p-değeri": [bp_result[1]]
        })
        bp_df.to_excel(writer, sheet_name="BP Testi", index=False)

        # Wooldridge
        wool_df = pd.DataFrame({
            "F İstatistiği": [wool_result[0]],
            "p-değeri": [wool_result[1]]
        })
        wool_df.to_excel(writer, sheet_name="Wooldridge Testi", index=False)

    output.seek(0)
    return output
