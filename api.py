from fastapi import FastAPI, Response
import mlflow
import pandas as pd
import json

#!pip install uvicorn

#Referência: https://anderfernandez.com/en/blog/how-to-create-api-python/

#################----INSTRUÇÕES----###############################
#
# INICIAR O MLFLOW USANDO O COMANDO ABAIXO NA PASTA DO PROJETO:
#
#     .\venv\scripts\activate
#
#     mlflow ui
#
#
# EM OUTRO TERMINAL, INICIAR O SERVIDOR USANDO O COMANDO NA PASTA DO PROJETO:
#
#     uvicorn api:app --reload 
#
# onde, api é o nome do programa em python da API (api.py)
#
###################################################################


#Conectando MLFlow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Instancia o Flask
app = FastAPI()

# API Modelo Desempenho Estudante

# Acesso via http://aplicacao/api/v1/modelo_estudante_escola/predict

# Cria um método GET
@app.get('/modelo_estudante_escola/predict', response_model=str)
def modelo_desempenho_predict(texp: float, horas: int):
    
    #Coletando o modelo em produção
    modelo_estagio_producao_predict = mlflow.sklearn\
        .load_model('models:/modelo_estudante_escola/production')
    
    #Criando DataFrame para ser consumido pelo Modelo
    dados_previsao = pd.DataFrame({'texp':[texp], 
                                   'horas':[horas]})
    
    resultado = modelo_estagio_producao_predict(dados_previsao)
    resultado = f'O desempenho previsto para este estudante é de: {resultado[0]:.2f}'
    
    return Response(content=resultado, media_type="text/plain")


# API Modelo Previsão Doença

# Acesso via http://aplicacao/api/v1/modelo_logistico_doenca/predict

# Cria um método GET
@app.get('/modelo_logistico_doenca/predict')
def modelo_doenca_predict(male: int, age: int, cigsPerDay: float):
    
    #Coletando o modelo em produção
    modelo_estagio_producao_predict = mlflow.sklearn\
        .load_model('models:/modelo_logistico_doenca/production')
    
    #Criando DataFrame para ser consumido pelo Modelo
    dados_previsao = pd.DataFrame({'male':[male], 
                                   'age':[age], 
                                   'cigsPerDay':[cigsPerDay]})
    
    resultado = modelo_estagio_producao_predict(dados_previsao)
    resultado = f'A probabilidade de ter um problema cardíaco em 10 anos é de: {resultado[0]*100:.2f}%'
    
    return Response(content=resultado, media_type="text/plain")