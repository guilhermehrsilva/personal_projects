import pandas as pd
import numpy as np
import os
import joblib
import warnings
from prefect import task, flow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from lightgbm import LGBMClassifier

# Suprimir warnings repetitivos de feature names na calibração
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Diretorios
BASE_DIR = r"c:\Users\Guilherme Risson\.gemini\antigravity\scratch\personal_projects\16_05_2026"
SEED = 42

# --- 1. Definir as Tarefas Individuais ---

@task(name="Extração de Dados", retries=2, retry_delay_seconds=5)
def extrair_dados_novos():
    print("Carregando CSVs locais de leads e eventos...")
    leads_path = os.path.join(BASE_DIR, 'leads_db.csv')
    eventos_path = os.path.join(BASE_DIR, 'eventos_db.csv')
    df_leads = pd.read_csv(leads_path)
    df_eventos = pd.read_csv(eventos_path)
    return df_leads, df_eventos

@task(name="Engenharia de Features")
def preparar_features(df_leads, df_eventos):
    print("Processando features e janelas temporais...")
    hoje = pd.to_datetime('2026-05-20')
    df_leads['data_criacao'] = pd.to_datetime(df_leads['data_criacao'])
    df_eventos['data_evento'] = pd.to_datetime(df_eventos['data_evento'])
    
    df_eventos['dias_atras'] = (hoje - df_eventos['data_evento']).dt.days
    df_eventos['evento_7d'] = (df_eventos['dias_atras'] <= 7).astype(int)
    df_eventos['evento_14d'] = (df_eventos['dias_atras'] <= 14).astype(int)
    df_eventos['evento_30d'] = (df_eventos['dias_atras'] <= 30).astype(int)
    
    features_temporais = df_eventos.groupby('lead_id').agg(
        engajamento_7d=('evento_7d', 'sum'),
        engajamento_14d=('evento_14d', 'sum'),
        engajamento_30d=('evento_30d', 'sum'),
        total_eventos_historico=('tipo_evento', 'count')
    ).reset_index()
    
    df_model = pd.merge(df_leads, features_temporais, on='lead_id', how='left')
    df_model[['engajamento_7d', 'engajamento_14d', 'engajamento_30d', 'total_eventos_historico']] = df_model[['engajamento_7d', 'engajamento_14d', 'engajamento_30d', 'total_eventos_historico']].fillna(0)
    df_model['dias_no_funil'] = (hoje - df_model['data_criacao']).dt.days
    
    # Target: 1 = Ganho, 0 = Aberto/Perdido
    df_model['target'] = (df_model['status_atual'] == 'Ganho').astype(int)
    
    numeric_features = ['engajamento_7d', 'engajamento_14d', 'engajamento_30d', 'total_eventos_historico', 'dias_no_funil']
    categorical_features = ['origem', 'porte_empresa']
    ALL_FEATURES = numeric_features + categorical_features
    
    X = df_model[ALL_FEATURES]
    y = df_model['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    print(f"Dados processados. Treino: {X_train.shape} | Teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test, ALL_FEATURES

@task(name="Treinar Modelo Champion")
def treinar_modelo(X_train, y_train, ALL_FEATURES):
    print("Treinando o LightGBM com busca de hiperparametros (GridSearchCV)...")
    numeric_features = ['engajamento_7d', 'engajamento_14d', 'engajamento_30d', 'total_eventos_historico', 'dias_no_funil']
    categorical_features = ['origem', 'porte_empresa']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )
    
    lgbm = LGBMClassifier(random_state=SEED, verbosity=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])
    
    # Grid otimizado: 12 combinações × 3 folds = 36 fits (vs 243 anterior)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 7],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [31]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Melhores parametros: {grid.best_params_}")
    
    best_pipeline = grid.best_estimator_
    print("Aplicando Calibracao Isotonica (3-fold)...")
    calibrated_model = CalibratedClassifierCV(best_pipeline, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    print("Calibracao concluida!")
    return calibrated_model

@task(name="Avaliar e Atualizar API")
def avaliar_e_salvar(modelo, X_test, y_test, ALL_FEATURES, threshold_antigo_auc=0.80):
    print("A avaliar a performance contra o baseline...")
    y_proba_test = modelo.predict_proba(X_test)[:, 1]
    auc_novo = roc_auc_score(y_test, y_proba_test)
    ap_novo = average_precision_score(y_test, y_proba_test)
    print(f"AUC do Novo Modelo: {auc_novo:.4f} (Baseline: {threshold_antigo_auc:.4f})")
    print(f"Average Precision: {ap_novo:.4f}")
    
    if auc_novo >= threshold_antigo_auc:
        print(f"Sucesso! Novo AUC ({auc_novo:.4f}) superou o baseline. Calculando threshold de ROI...")
        TICKET_MEDIO = 5000
        CUSTO_LIGACAO = 50
        thresholds = np.linspace(0.01, 0.99, 100)
        lucros = []
        for t in thresholds:
            y_pred_t = (y_proba_test >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
            lucro = (tp * TICKET_MEDIO) - ((tp + fp) * CUSTO_LIGACAO)
            lucros.append(lucro)
        
        best_threshold = thresholds[np.argmax(lucros)]
        max_lucro = max(lucros)
        print(f"Threshold Otimo de ROI: {best_threshold:.4f} (~{best_threshold*100:.1f}%)")
        print(f"Lucro Maximo Projetado: R$ {max_lucro:,.2f}")
        
        joblib.dump(modelo, os.path.join(BASE_DIR, 'modelo_lead_scoring.pkl'))
        joblib.dump(ALL_FEATURES, os.path.join(BASE_DIR, 'features_modelo.pkl'))
        joblib.dump(best_threshold, os.path.join(BASE_DIR, 'threshold_otimo.pkl'))
        print("[OK] Modelo exportado com sucesso!")
        return "Deploy Autorizado"
    else:
        print(f"Falha! O novo modelo piorou ({auc_novo:.4f} < {threshold_antigo_auc:.4f}). Abortando deploy.")
        return "Deploy Abortado"

# --- 2. Definir o Fluxo Principal (Orquestrador) ---

@flow(name="Retreino Mensal - Lead Scoring B2B", description="Pipeline automatico de atualizacao do modelo")
def orquestrador_mlops():
    df_leads, df_eventos = extrair_dados_novos()
    X_train, X_test, y_train, y_test, ALL_FEATURES = preparar_features(df_leads, df_eventos)
    modelo_fresco = treinar_modelo(X_train, y_train, ALL_FEATURES)
    resultado = avaliar_e_salvar(modelo_fresco, X_test, y_test, ALL_FEATURES)
    print(f"Status Final do Pipeline: {resultado}")

if __name__ == "__main__":
    # Rodar o pipeline manualmente
    orquestrador_mlops()