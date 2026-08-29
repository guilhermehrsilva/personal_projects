from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import uvicorn

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Carregar o modelo e configurações (arranca ao iniciar o servidor)
try:
    modelo = joblib.load(os.path.join(BASE_DIR, 'modelo_lead_scoring.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, 'features_modelo.pkl'))
    threshold_otimo = joblib.load(os.path.join(BASE_DIR, 'threshold_otimo.pkl'))
except Exception as e:
    raise RuntimeError(f"Erro ao carregar ficheiros do modelo: {e}. Corre o notebook primeiro.")

# 2. Definir o esquema de dados de entrada com Pydantic
class LeadInput(BaseModel):
    engajamento_7d: int = Field(..., description="Eventos nos últimos 7 dias")
    engajamento_14d: int = Field(..., description="Eventos nos últimos 14 dias")
    engajamento_30d: int = Field(..., description="Eventos nos últimos 30 dias")
    total_eventos_historico: int = Field(..., description="Total de interações")
    dias_no_funil: int = Field(..., description="Dias desde a criação do lead")
    origem: str = Field(..., description="Ex: Inbound, Outbound, Eventos")
    porte_empresa: str = Field(..., description="Ex: Pequena, Média, Grande, Enterprise")

# 3. Inicializar a API
app = FastAPI(title="Lead Scoring B2B API", version="1.0")

# PERMISSÃO DE CORS: Libera o acesso para o seu dashboard standalone consumir os dados
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite qualquer origem para o nosso teste local
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "API de ML Online e Saudável"}

@app.post("/predict")
def prever_score(lead: LeadInput):
    try:
        # Converter o input JSON num DataFrame (1 linha)
        df_input = pd.DataFrame([lead.dict()])
        
        # Garantir a ordem correta das features
        df_input = df_input[features]
        
        # Calcular probabilidade
        probabilidade = modelo.predict_proba(df_input)[0, 1]
        
        # Definir a ação com base no threshold financeiro otimizado
        vai_converter = bool(probabilidade >= threshold_otimo)
        
        # Segmentação Comercial (Tiers)
        if probabilidade >= 0.8:
            tier, acao = "Crítico", "Ligar HOJE — alto potencial"
        elif probabilidade >= 0.6:
            tier, acao = "Quente", "Agendar reunião esta semana"
        elif probabilidade >= 0.3:
            tier, acao = "Morno", "Nutrir com conteúdo + follow-up 15d"
        else:
            tier, acao = "Frio", "Manter em automação"

        return {
            "score_probabilidade": round(float(probabilidade), 4),
            "tier": tier,
            "acao_recomendada": acao,
            "ultrapassou_threshold_roi": vai_converter
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)