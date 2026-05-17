import pandas as pd
import joblib
from prefect import task, flow
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# --- 1. Definir as Tarefas Individuais ---

@task(name="Extração de Dados", retries=2, retry_delay_seconds=5)
def extrair_dados_novos():
    print("A simular extração do Data Warehouse...")
    # Em produção, aqui entraria a query SQL ou o consumo da API.
    # Para o pipeline não quebrar, vamos carregar os mesmos dados da simulação de Drift
    # (assumindo que exportaste um CSV ou conectas à base de dados real)
    # Aqui retornamos um DataFrame fictício apenas para a estrutura do pipeline rodar
    return pd.DataFrame() 

@task(name="Engenharia de Features")
def preparar_features(df_bruto):
    print("A calcular janelas temporais de 7, 14 e 30 dias...")
    # Aqui entra o código de transformação do pandas
    return pd.DataFrame(), pd.Series() # Retorna X e y

@task(name="Treinar Modelo Champion")
def treinar_modelo(X_train, y_train):
    print("A treinar o LightGBM com dados frescos...")
    
    # Recriar o pipeline
    numeric_features = ['engajamento_7d', 'engajamento_14d', 'engajamento_30d', 'total_eventos_historico', 'dias_no_funil']
    categorical_features = ['origem', 'porte_empresa']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])
    
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])
    
    clf_calibrado = CalibratedClassifierCV(pipeline, method='isotonic', cv=5)
    
    # Simular o treino (comentado para não quebrar sem dados reais)
    # clf_calibrado.fit(X_train, y_train) 
    
    return clf_calibrado

@task(name="Avaliar e Atualizar API")
def avaliar_e_salvar(modelo, X_test, y_test, threshold_antigo_auc=0.80):
    print("A validar a performance contra o baseline...")
    # Simulação da métrica do novo modelo
    auc_novo = 0.83 
    
    if auc_novo >= threshold_antigo_auc:
        print(f"Sucesso! Novo AUC ({auc_novo}) superou o anterior. A exportar .pkl...")
        # joblib.dump(modelo, 'modelo_lead_scoring.pkl')
        return "Deploy Autorizado"
    else:
        print("Falha! O novo modelo piorou. Abortando deploy.")
        return "Deploy Abortado"

# --- 2. Definir o Fluxo Principal (Orquestrador) ---

@flow(name="Retreino Mensal - Lead Scoring B2B", description="Pipeline automático de atualização do modelo")
def orquestrador_mlops():
    # A magia do Prefect acontece aqui: ele gere a ordem, os logs e as falhas
    df_bruto = extrair_dados_novos()
    X, y = preparar_features(df_bruto)
    
    # Num cenário real faríamos o train_test_split aqui
    modelo_fresco = treinar_modelo(X, y)
    
    resultado = avaliar_e_salvar(modelo_fresco, X, y)
    print(f"Status Final do Pipeline: {resultado}")

if __name__ == "__main__":
    # Rodar o pipeline manualmente
    orquestrador_mlops()