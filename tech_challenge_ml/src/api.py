from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# 1. Configuração de Caminhos (Resolução dinâmica)
CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(CAMINHO_BASE, 'models', 'modelo_churn.joblib')

# 2. Inicialização da Aplicação
app = FastAPI(
    title="API de Retenção de Clientes (Churn)",
    description="Motor de inferência de Machine Learning para prever a probabilidade de cancelamento.",
    version="1.0.0"
)

# 3. Carregamento do Modelo em Memória (Arranque da aplicação)
try:
    modelo = joblib.load(CAMINHO_MODELO)
    # Capturar a assinatura exata das colunas (features) que o modelo espera
    colunas_requeridas = modelo.feature_names_in_
except Exception as e:
    modelo = None
    colunas_requeridas = []
    print(f"⚠️ Aviso: Não foi possível carregar o modelo. Detalhe: {e}")

# 4. Definição do Contrato de Dados (Payload de Entrada)
class DadosCliente(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # Em iterações futuras, pode adicionar os restantes campos categóricos aqui

# 5. Rotas da API
@app.get("/")
def health_check():
    """Endpoint para monitoramento de CI/CD e verificação de saúde da API."""
    return {
        "status": "Operacional",
        "modelo_carregado": modelo is not None
    }

@app.post("/predict")
def prever_churn(cliente: DadosCliente):
    """
    Recebe os dados do cliente, processa as features e devolve a propensão ao churn.
    """
    if modelo is None:
        raise HTTPException(
            status_code=503, 
            detail="Serviço indisponível: O modelo de Machine Learning não foi encontrado."
        )

    # Preencher todas as colunas exigidas pelo modelo com 0 por defeito
    dados_entrada = {col: 0 for col in colunas_requeridas}
    
    # Injetar os dados reais fornecidos no JSON da requisição
    if 'tenure' in dados_entrada: 
        dados_entrada['tenure'] = cliente.tenure
    if 'MonthlyCharges' in dados_entrada: 
        dados_entrada['MonthlyCharges'] = cliente.MonthlyCharges
    if 'TotalCharges' in dados_entrada: 
        dados_entrada['TotalCharges'] = cliente.TotalCharges
    
    # Converter para o formato tabular esperado pelo Scikit-Learn
    df_cliente = pd.DataFrame([dados_entrada])
    
    # Garantir o alinhamento estrito das colunas
    df_cliente = df_cliente[colunas_requeridas]
    
    # Executar a inferência matemática
    previsao = int(modelo.predict(df_cliente)[0])
    probabilidade = modelo.predict_proba(df_cliente)[0][1]
    
    status_final = "Risco de Cancelamento (Atuar Imediatamente)" if previsao == 1 else "Cliente Retido"
    
    return {
        "churn_previsao": previsao,
        "status_cliente": status_final,
        "probabilidade_cancelamento": round(probabilidade, 4)
    }
