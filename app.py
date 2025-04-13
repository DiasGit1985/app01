import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Previsão de Vendas ERP (HTML)", layout="wide")
st.title("📈 Previsão de Vendas direto do ERP (HTML com SQL)")

st.markdown("""
Este app aceita arquivos HTML exportados do ERP, mesmo que contenham o comando SQL acima da tabela.

A tabela válida deve conter:
- Dt_Movimento
- Cd_material
- Descricao_Material
- Quantidade
- Cd_tp_operacao
- Tipo_movimento
- Cd_sub_grupo
- Descricao_Subgrupo
""")

arquivo = st.file_uploader("📤 Envie seu arquivo HTML exportado do ERP", type=["html"])

if arquivo:
    try:
        tabelas = pd.read_html(arquivo)

        # Encontrar a tabela que contém a coluna Descricao_Subgrupo
        df = None
        for t in tabelas:
            if "Descricao_Subgrupo" in t.columns:
                df = t
                break

        if df is None:
            st.error("Nenhuma tabela válida com a coluna 'Descricao_Subgrupo' foi encontrada no arquivo.")
        else:
            st.subheader("Pré-visualização dos dados válidos")
            st.dataframe(df.head())

            # Renomear para os nomes padrão do modelo
            df = df.rename(columns={
                "Dt_Movimento": "Data",
                "Quantidade": "Quantidade Vendida",
                "Descricao_Material": "Produto",
                "Descricao_Subgrupo": "Subgrupo",
                "Cd_material": "Codigo"
            })

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