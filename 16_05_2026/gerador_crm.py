import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import uvicorn
from fastapi import FastAPI

# Configurações iniciais
np.random.seed(42)
n_leads = 10000
hoje = datetime(2026, 5, 20)  # Data de referência do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def gerar_dados():
    print("Gerando banco de dados expandido de leads (10.000 registros)...")
    
    # 1. Gerar Tabela de Leads
    lead_ids = [f'L{str(i).zfill(5)}' for i in range(1, n_leads + 1)]
    origens = ['Inbound', 'Outbound', 'Parceiros', 'Eventos']
    portes = ['Pequena', 'Média', 'Grande', 'Enterprise']

    df_leads = pd.DataFrame({
        'lead_id': lead_ids,
        'data_criacao': [hoje - timedelta(days=np.random.randint(10, 180)) for _ in range(n_leads)],
        'origem': np.random.choice(origens, n_leads, p=[0.5, 0.3, 0.1, 0.1]),
        'porte_empresa': np.random.choice(portes, n_leads, p=[0.4, 0.35, 0.15, 0.1]),
        'status_atual': np.random.choice(['Aberto', 'Ganho', 'Perdido'], n_leads, p=[0.5, 0.2, 0.3])
    })

    # 2. Gerar Tabela de Eventos (Log Transacional)
    tipos_evento = ['email_aberto', 'email_clicado', 'site_visita_pricing', 'download_ebook', 'reuniao_agendada']
    pesos_eventos = [0.4, 0.2, 0.15, 0.15, 0.1]
    eventos = []

    for _, lead in df_leads.iterrows():
        # Leads 'Ganhos' tendem a ter muito mais interações/eventos
        n_eventos = np.random.randint(2, 18) if lead['status_atual'] == 'Ganho' else np.random.randint(0, 8)
        
        for _ in range(n_eventos):
            dias_desde_criacao = (hoje - lead['data_criacao']).days
            if dias_desde_criacao > 0:
                dias_atras = np.random.randint(0, dias_desde_criacao)
                data_evento = hoje - timedelta(days=dias_atras)
                
                eventos.append({
                    'lead_id': lead['lead_id'],
                    'data_evento': data_evento,
                    'tipo_evento': np.random.choice(tipos_evento, p=pesos_eventos)
                })

    df_eventos = pd.DataFrame(eventos)
    df_eventos = df_eventos.sort_values(by=['data_evento']).reset_index(drop=True)
    
    # Converter datas para strings formatadas antes de salvar ou retornar
    df_leads['data_criacao'] = df_leads['data_criacao'].dt.strftime('%Y-%m-%d')
    df_eventos['data_evento'] = df_eventos['data_evento'].dt.strftime('%Y-%m-%d')

    # Salvar localmente em CSV para leitura direta e rápida
    leads_path = os.path.join(BASE_DIR, 'leads_db.csv')
    eventos_path = os.path.join(BASE_DIR, 'eventos_db.csv')
    
    df_leads.to_csv(leads_path, index=False, encoding='utf-8')
    df_eventos.to_csv(eventos_path, index=False, encoding='utf-8')
    
    print(f"Banco de dados gerado! Leads: {len(df_leads)} | Eventos: {len(df_eventos)}")
    print(f"Salvo em CSV:\n - {leads_path}\n - {eventos_path}")
    
    return df_leads, df_eventos

# Gerar dados no momento da importação/execução se os CSVs não existirem
leads_file = os.path.join(BASE_DIR, 'leads_db.csv')
eventos_file = os.path.join(BASE_DIR, 'eventos_db.csv')

if not os.path.exists(leads_file) or not os.path.exists(eventos_file):
    df_leads, df_eventos = gerar_dados()
else:
    print("CSVs existentes encontrados. Carregando dados para a API...")
    df_leads = pd.read_csv(leads_file, encoding='utf-8')
    df_eventos = pd.read_csv(eventos_file, encoding='utf-8')

# --- API FASTAPI (MOCK CRM) ---
app = FastAPI(title="Mock CRM API B2B - Porta 8001")

@app.get("/api/v1/leads")
def get_leads(skip: int = 0, limit: int = 100):
    return {"total": len(df_leads), "data": df_leads.iloc[skip : skip + limit].to_dict(orient="records")}

@app.get("/api/v1/eventos")
def get_eventos(skip: int = 0, limit: int = 500):
    return {"total": len(df_eventos), "data": df_eventos.iloc[skip : skip + limit].to_dict(orient="records")}

if __name__ == "__main__":
    # Permite rodar o gerador separadamente e iniciar a API do CRM na porta 8001
    print("Iniciando API Mock CRM na porta 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
