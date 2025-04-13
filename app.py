import streamlit as st
import pandas as pd
from prophet import Prophet
import unicodedata
import io
from fpdf import FPDF

st.set_page_config(page_title="Previsão de Compras - Simplificado", layout="wide")
st.title("📦 Previsão de Vendas para Requisição de Compras")

def normalizar(col):
    return unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8").lower().replace(" ", "").replace("_", "")

colunas_esperadas = {
    "dtmovimento": "Data",
    "cdmaterial": "Codigo",
    "descricaomaterial": "Produto",
    "quantidade": "Quantidade Vendida",
    "descricaosubgrupo": "Subgrupo",
    "tipomovimento": "Tipo"
}

arquivo = st.file_uploader("📤 Envie seu arquivo HTML exportado do ERP", type=["html"])

if arquivo:
    try:
        tabelas = pd.read_html(arquivo, header=1)
        df = tabelas[0]

        st.subheader("📑 Pré-visualização da planilha principal (dados brutos)")
        st.dataframe(df.head())

        renomear = {}
        for col in df.columns:
            nome_normalizado = normalizar(col)
            if nome_normalizado in colunas_esperadas:
                renomear[col] = colunas_esperadas[nome_normalizado]
        df = df.rename(columns=renomear)

        faltando = [v for v in colunas_esperadas.values() if v not in df.columns]
        if faltando:
            st.error(f"Colunas ausentes no arquivo: {', '.join(faltando)}")
        else:
            df = df[df['Tipo'] == 'S']
            df['Data'] = pd.to_datetime(df['Data'])

            subgrupos = df['Subgrupo'].unique()
            subgrupo_selecionado = st.selectbox("Selecione o subgrupo", subgrupos)
            meses = st.slider("Quantos meses deseja prever?", 1, 12, 3)

            df_filtrado = df[df['Subgrupo'] == subgrupo_selecionado]
            produtos_codigos = df_filtrado[['Produto', 'Codigo']].drop_duplicates()
            produtos_codigos['Identificador'] = produtos_codigos['Produto'] + " (Cód. " + produtos_codigos['Codigo'].astype(str) + ")"

            produto_selecionado = st.selectbox("Selecione o produto para exibir as previsões", produtos_codigos['Identificador'])

            linha = produtos_codigos[produtos_codigos['Identificador'] == produto_selecionado].iloc[0]
            produto = linha['Produto']
            codigo = linha['Codigo']

            st.markdown(f"### 📊 Previsão para: **{produto}** (Código: {codigo})")

            df_item = df_filtrado[df_filtrado['Codigo'] == codigo][['Data', 'Quantidade Vendida']].copy()
            df_item = df_item.rename(columns={"Data": "ds", "Quantidade Vendida": "y"})
            df_item['ds'] = pd.to_datetime(df_item['ds'])

            modelo = Prophet()
            modelo.fit(df_item)
            futuro = modelo.make_future_dataframe(periods=meses, freq='M')
            previsao = modelo.predict(futuro)

            previsoes_final = previsao[['ds', 'yhat']].tail(meses).copy()
            previsoes_final = previsoes_final.rename(columns={
                'ds': 'Mês Previsto',
                'yhat': 'Quantidade Prevista'
            })
            previsoes_final['Mês Previsto'] = previsoes_final['Mês Previsto'].dt.to_period('M').astype(str)

            # Comparativo com média dos últimos 3 meses reais
            ultimos_meses_real = df_item.sort_values(by="ds").tail(90)
            media_ultimos = ultimos_meses_real['y'].mean()
            media_previsao = previsoes_final['Quantidade Prevista'].mean()

            delta = media_previsao - media_ultimos
            cor = "🟢 Aumentando" if delta > 0 else "🔴 Reduzindo" if delta < 0 else "🟡 Estável"

            st.metric("Tendência de Vendas", value=f"{cor}", delta=f"{delta:.1f} unidades")

            st.dataframe(previsoes_final)

            # Total previsto para subgrupo
            total_geral = 0
            for cod in produtos_codigos['Codigo'].unique():
                df_temp = df_filtrado[df_filtrado['Codigo'] == cod][['Data', 'Quantidade Vendida']].copy()
                df_temp = df_temp.rename(columns={"Data": "ds", "Quantidade Vendida": "y"})
                df_temp['ds'] = pd.to_datetime(df_temp['ds'])

                if len(df_temp) >= 2:
                    modelo = Prophet()
                    modelo.fit(df_temp)
                    futuro = modelo.make_future_dataframe(periods=meses, freq='M')
                    previsao_temp = modelo.predict(futuro)
                    total_geral += previsao_temp['yhat'].tail(meses).sum()

            st.info(f"📦 Previsão total para o subgrupo **{subgrupo_selecionado}** nos próximos {meses} meses: **{int(total_geral)} unidades**")

            # Exportar Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                previsoes_final.to_excel(writer, index=False, sheet_name="Previsao_Selecionado")

            st.download_button(
                label="📥 Baixar previsão deste item em Excel",
                data=buffer.getvalue(),
                file_name=f"previsao_{codigo}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # PDF para o comprador
            if st.button("📤 Gerar PDF para enviar ao comprador"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.cell(200, 10, f"Previsão de Vendas - Produto {produto} (Código: {codigo})", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", "", 10)
                for i, row in previsoes_final.iterrows():
                    pdf.cell(200, 8, f"{row['Mês Previsto']}: {row['Quantidade Prevista']:.0f} unidades", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", "I", 9)
                pdf.cell(200, 10, f"Tendência: {cor} | Total previsto no subgrupo: {int(total_geral)} unidades", ln=True)
                pdf_output = io.BytesIO()
                pdf.output(pdf_output)
                st.download_button(
                    label="📄 Baixar PDF para o comprador",
                    data=pdf_output.getvalue(),
                    file_name=f"Previsao_Comprador_{codigo}.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo HTML: {e}")