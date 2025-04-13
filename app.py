import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão de Vendas ERP (HTML)", layout="wide")
st.title("📈 Previsão de Vendas direto do ERP (arquivo HTML)")

st.markdown("""
Envie o arquivo HTML exportado do ERP contendo os dados de movimentação de estoque.  
A planilha deve conter as colunas: `Dt_Movimento`, `Quantidade`, `Cd_material`, `Descricao_Material`, `Descricao_Subgrupo`  
O sistema converterá e padronizará os campos automaticamente.
""")

arquivo = st.file_uploader("📤 Envie seu arquivo HTML exportado", type=["html"])

if arquivo:
    try:
        # Lê todas as tabelas contidas no HTML
        tabelas = pd.read_html(arquivo)
        df = tabelas[0]  # Assume que a primeira tabela é a correta

        st.subheader("Pré-visualização dos dados brutos")
        st.dataframe(df.head())

        # Renomear as colunas conforme esperado pelo modelo
        df = df.rename(columns={
            "Dt_Movimento": "Data",
            "Quantidade": "Quantidade Vendida",
            "Descricao_Material": "Produto",
            "Descricao_Subgrupo": "Subgrupo",
            "Cd_material": "Codigo"
        })

        # Filtros
        subgrupos = df['Subgrupo'].unique()
        subgrupo_selecionado = st.selectbox("Selecione o subgrupo", subgrupos)

        meses = st.slider("Quantos meses deseja prever?", 1, 12, 3)

        df_filtrado = df[df['Subgrupo'] == subgrupo_selecionado]

        # Agrupar por produto + código
        produtos_codigos = df_filtrado[['Produto', 'Codigo']].drop_duplicates()

        for _, row in produtos_codigos.iterrows():
            produto = row['Produto']
            codigo = row['Codigo']
            st.markdown(f"### 📦 Produto: {produto} (Código: {codigo})")

            df_prod = df_filtrado[df_filtrado['Codigo'] == codigo][['Data', 'Quantidade Vendida']].copy()
            df_prod = df_prod.rename(columns={"Data": "ds", "Quantidade Vendida": "y"})
            df_prod['ds'] = pd.to_datetime(df_prod['ds'])

            modelo = Prophet()
            modelo.fit(df_prod)

            futuro = modelo.make_future_dataframe(periods=meses, freq='M')
            previsao = modelo.predict(futuro)

            fig1 = modelo.plot(previsao)
            st.pyplot(fig1)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo HTML: {e}")