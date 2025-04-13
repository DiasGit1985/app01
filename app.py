import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão de Vendas por Subgrupo", layout="wide")
st.title("📈 Previsão de Vendas com IA por Subgrupo de Produtos")

st.markdown("""
Envie uma planilha com colunas:
- `Data`
- `Quantidade Vendida`
- `Produto`
- `Subgrupo` (ou Código do Subgrupo)

Depois selecione o subgrupo e a quantidade de meses a prever.
""")

arquivo = st.file_uploader("📤 Envie sua planilha Excel", type=["xlsx"])

if arquivo:
    try:
        df = pd.read_excel(arquivo)
        st.subheader("Pré-visualização dos dados")
        st.dataframe(df.head())

        # Filtro de subgrupo
        subgrupos = df['Subgrupo'].unique()
        subgrupo_selecionado = st.selectbox("Selecione o subgrupo", subgrupos)

        # Filtro de meses para previsão
        meses = st.slider("Quantos meses deseja prever?", 1, 12, 3)

        # Filtrar pelo subgrupo
        df_filtrado = df[df['Subgrupo'] == subgrupo_selecionado]

        # Obter lista de produtos dentro do subgrupo
        produtos = df_filtrado['Produto'].unique()

        for produto in produtos:
            st.markdown(f"### 📦 Produto: {produto}")
            df_prod = df_filtrado[df_filtrado['Produto'] == produto][['Data', 'Quantidade Vendida']].copy()
            df_prod = df_prod.rename(columns={"Data": "ds", "Quantidade Vendida": "y"})
            df_prod['ds'] = pd.to_datetime(df_prod['ds'])

            modelo = Prophet()
            modelo.fit(df_prod)

            futuro = modelo.make_future_dataframe(periods=meses, freq='M')
            previsao = modelo.predict(futuro)

            fig1 = modelo.plot(previsao)
            st.pyplot(fig1)

    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")