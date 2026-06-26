from fastapi.testclient import TestClient
from src.api import app

# Inicializar o cliente de testes do FastAPI
client = TestClient(app)

def test_health_check():
    """Garante que a API está online e a responder."""
    response = client.get("/")
    
    assert response.status_code == 200
    dados = response.json()
    assert "status" in dados
    assert dados["status"] == "Operacional"
    # Verifica se o modelo foi carregado corretamente na memória
    assert dados["modelo_carregado"] is True

def test_prever_churn_sucesso():
    """Simula o envio de um cliente válido e verifica a estrutura da resposta."""
    payload = {
        "tenure": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0
    }
    
    response = client.post("/predict", json=payload)
    
    # A API deve aceitar o pedido (Status 200)
    assert response.status_code == 200
    
    # Validar se o dicionário JSON de retorno tem as chaves de negócio corretas
    dados = response.json()
    assert "churn_previsao" in dados
    assert "status_cliente" in dados
    assert "probabilidade_cancelamento" in dados

def test_prever_churn_erro_validacao():
    """Garante que a API recusa pedidos com campos obrigatórios em falta."""
    # Faltam o MonthlyCharges e o TotalCharges
    payload = {
        "tenure": 12
    }
    
    response = client.post("/predict", json=payload)
    
    # O Pydantic deve intercetar a falha e devolver um erro 422 (Unprocessable Entity)
    assert response.status_code == 422
