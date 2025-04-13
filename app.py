import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão de Vendas ERP (HTML fixo)", layout="wide")
st.title("📈 Previsão de Vendas direto do ERP (HTML com cabeçalho fixo)")

st.markdown("""
Este app foi ajustado para arquivos HTML exportados do ERP que contêm:
- A primeira linha com o comando SQL
- A segunda linha com os títulos das colunas
- A partir da terceira linha, os dados de fato
""")

def normalizar_nome(nome_coluna):
    import unicodedata
    return unicodedata.normalize("NFKD", nome_coluna)                     .encode("ASCII", "ignore")                     .decode("utf-8")                     .replace(" ", "")                     .lower()

arquivo = st.file_uploader("📤 Envie seu arquivo HTML exportado do ERP", type=["html"])

if arquivo:
    try:
        # Força o header na segunda linha (index 1), ou seja, pula a primeira linha (com o SQL)
        tabelas = pd.read_html(arquivo, header=1)
        df = tabelas[0]

        st.subheader("Pré-visualização da tabela principal")
        st.dataframe(df.head())

        # Renomear colunas ignorando acentos e variações
        mapeamento = {}
        for col in df.columns:
            nome = normalizar_nome(col)
            if nome == "dt_movimento":
                mapeamento[col] = "Data"
            elif nome == "quantidade":
                mapeamento[col] = "Quantidade Vendida"
            elif nome == "descricaomaterial":
                mapeamento[col] = "Produto"
            elif nome == "descricaosubgrupo":
                mapeamento[col] = "Subgrupo"
            elif nome == "cd_material":
                mapeamento[col] = "Codigo"

        df = df.rename(columns=mapeamento)

        subgrupos = df['Subgrupo'].unique()
        subgrupo_selecionado = st.selectbox("Selecione o subgrupo", subgrupos)
        meses = st.slider("Quantos meses deseja prever?", 1, 12, 3)

        df_filtrado = df[df['Subgrupo'] == subgrupo_selecionado]
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