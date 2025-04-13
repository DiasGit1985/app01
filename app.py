import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão de Vendas", layout="wide")
st.title("📈 Previsão de Vendas com IA")

st.markdown("""
Esta ferramenta permite subir uma planilha com os dados históricos de vendas para gerar a previsão dos próximos 3 meses.  
A planilha deve conter **duas colunas**:
- `Data` (formato ano-mês, ex: 2023-01)
- `Quantidade Vendida`
""")

# Upload
arquivo = st.file_uploader("Envie sua planilha Excel (.xlsx)", type=["xlsx"])

if arquivo:
    try:
        df = pd.read_excel(arquivo)

        st.subheader("Pré-visualização dos dados")
        st.dataframe(df.head())

        # Preparar dados para o Prophet
        df = df.rename(columns={"Data": "ds", "Quantidade Vendida": "y"})
        df["ds"] = pd.to_datetime(df["ds"])

        modelo = Prophet()
        modelo.fit(df)

        futuro = modelo.make_future_dataframe(periods=3, freq='M')
        previsao = modelo.predict(futuro)

        st.subheader("📊 Gráfico da Previsão")
        fig1 = modelo.plot(previsao)
        st.pyplot(fig1)

        st.subheader("📉 Componentes da Previsão")
        fig2 = modelo.plot_components(previsao)
        st.pyplot(fig2)

        # Mostrar a tabela com os valores futuros
        df_futuro = previsao[previsao["ds"] > df["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
        df_futuro.columns = ["Data Prevista", "Estimado", "Limite Inferior", "Limite Superior"]
        df_futuro = df_futuro.round(0).astype({"Estimado": int, "Limite Inferior": int, "Limite Superior": int})
        st.subheader("📄 Tabela de Previsões Futuras")
        st.dataframe(df_futuro)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")