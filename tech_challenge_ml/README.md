# Tech Challenge - Machine Learning Engineering (Fase 1)

## Objetivo do Projeto
Este projeto tem como objetivo construir e produtizar um modelo de Machine Learning para prever o cancelamento de clientes (Churn) em uma empresa de telecomunicações.

## Estrutura do Projeto
- `/data`: Contém a base de dados original (`Telco-Customer-Churn.csv`).
- `/models`: Contém o modelo treinado exportado (`modelo_churn.joblib`).
- `01_construcao_do_modelo.ipynb`: Notebook com a limpeza de dados, treino do modelo e simulação da API.
- `requirements.txt`: Lista de dependências necessárias para rodar o projeto.

## Como Executar a API Localmente
1. Instale as dependências: `pip install -r requirements.txt`
2. Execute o servidor: `uvicorn main:app --reload`
3. Acesse a documentação interativa em: `http://127.0.0.1:8000/docs`

## Modelo Utilizado
- Algoritmo: Random Forest Classifier
- Métrica Principal: Acurácia
