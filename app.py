import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import unicodedata

st.set_page_config(page_title="Previsão de Vendas ERP (Final Robusta)", layout="wide")
st.title("📈 Previsão de Vendas direto do ERP (HTML final robusto)")

st.markdown("""
Este app foi preparado para arquivos HTML exportados do ERP, com:
- Linha 1 contendo o SQL (ignorada)
- Linha 2 com os títulos das colunas
- Linha 3 em diante com os dados
- Nomes de colunas com ou sem acento são tratados automaticamente
""")

def normalizar(col):
    return unicodedata.normalize("NFKD", col).encode("ASCII", "ignore").decode("utf-8").lower().replace(" ", "").replace("_", "")

colunas_esperadas = {
    "dtmovimento": "Data",
    "cdmaterial": "Codigo",
    "descricaomaterial": "Produto",
    "quantidade": "Quantidade Vendida",
    "descricaosubgrupo": "Subgrupo"
}

arquivo = st.file_uploader("📤 Envie seu arquivo HTML exportado do ERP", type=["html"])

if arquivo:
    try:
        # Força o cabeçalho na segunda linha (pula o SQL)
        tabelas = pd.read_html(arquivo, header=1)
        df = tabelas[0]

        st.subheader("Pré-visualização da tabela principal")
        st.dataframe(df.head())

        # Normalizar colunas e aplicar renomeação se possível
        renomear = {}
        for col in df.columns:
            nome_normalizado = normalizar(col)
            if nome_normalizado in colunas_esperadas:
                renomear[col] = colunas_esperadas[nome_normalizado]

        df = df.rename(columns=renomear)

        # Verificar se todas as colunas essenciais foram encontradas
        faltando = [v for v in colunas_esperadas.values() if v not in df.columns]
        if faltando:
            st.error(f"Colunas ausentes no arquivo: {', '.join(faltando)}")
        else:
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