# 🚀 Lead Scoring B2B — Plataforma Comercial de MLOps

Esta plataforma é um ecossistema completo de **Machine Learning em Produção (MLOps)** projetado para identificar, priorizar e converter leads B2B de alto valor comercial. 

O projeto evoluiu de uma simples análise exploratória num Jupyter Notebook para uma **arquitetura robusta de produção** com uma API em tempo real (FastAPI), contentorização (Docker), orquestração de retreino (Prefect) e um simulador comercial interativo (Tailwind CSS + Chart.js).

---

## 📐 Arquitetura do Ecossistema

```
 ┌──────────────────────┐      ┌────────────────────────┐      ┌─────────────────────────┐
 │  Data Warehouse (DW) │ ───> │  Prefect Pipeline      │ ───> │  LightGBM + Calibration │
 │  (Dados Brutos B2B)  │      │  (Retreino Mensal)     │      │  (Ficheiros .pkl)       │
 └──────────────────────┘      └────────────────────────┘      └─────────────────────────┘
                                                                            │
                                                                            ▼
 ┌──────────────────────┐      ┌────────────────────────┐      ┌─────────────────────────┐
 │ Dashboard Comercial  │ <─── │ FastAPI API            │ <─── │ joblib (Carregamento    │
 │ (Tailwind/Chart.js)  │ (HTTP)│ (Docker Container)     │      │ em Memória Segura)      │
 └──────────────────────┘      └────────────────────────┘      └─────────────────────────┘
```

---

## 📂 Estrutura de Ficheiros do Projeto

*   **`analise_b2b.ipynb`**: Notebook de Data Science. Contém a análise exploratória de dados (EDA), modelagem experimental com Calibração Isotónica, análise de ROI e interpretabilidade usando **SHAP values**.
*   **`api_scoring.py`**: Servidor FastAPI de produção que expõe a rota `/predict`. Integra o modelo serializado com suporte a CORS para consumo standalone.
*   **`pipeline_retreino.py`**: Pipeline automático e resiliente usando **Prefect** para extração, validação, retreino mensal do LightGBM e autorização de deploy (Champion vs Baseline).
*   **`dashboard_comercial.html`**: Simulador de vendas interativo. Permite a simulação de leads com respostas imediatas da API (probabilidade, tier, ação comercial e ROI).
*   **`DockerFile`**: Receita Docker para empacotar a API FastAPI de forma leve (`python:3.10-slim`) com suporte nativo a operações matemáticas de árvores (`libgomp1`).
*   **`requirements.txt`**: Dependências Python do projeto com as versões mapeadas e fixadas.

---

## 🧠 Lógica Comercial: O Threshold Ótimo de ROI (~2.9%)

Ao contrário dos modelos acadêmicos que usam o ponto de corte padrão de `50%` de probabilidade, esta plataforma utiliza um **Threshold Ótimo de ROI (~2.9%)** calculado no notebook de negócios:
*   **Porquê?** Em vendas B2B complexas, o custo de uma ligação telefónica/email é drasticamente inferior ao retorno financeiro de uma conversão de sucesso (LTV).
*   **Ação:** Qualquer lead com probabilidade de conversão igual ou superior a **~2.9%** já justifica o acionamento da equipa comercial, gerando lucro líquido positivo (ROI Positivo). Leads abaixo deste threshold são mantidos em fluxos de nutrição automática (Bloqueados para o comercial).

---

## 🛠️ Como Executar o Projeto (Guia Passo a Passo)

### 1. Preparação do Ambiente Local
Crie o seu ambiente virtual e instale todas as dependências:
```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente (Windows)
.venv\Scripts\activate

# Instalar dependências completas
pip install -r requirements.txt
```

---

### 2. Rodar a API FastAPI (Duas Opções)

#### Opção A: Utilizando Docker (Produção)
Esta opção encapsula toda a aplicação numa imagem isolada de alto rendimento.
```bash
# 1. Construir a imagem Docker
docker build -t api-lead-scoring -f DockerFile .

# 2. Executar a API na porta 8080
docker run -p 8080:8080 api-lead-scoring
```

#### Opção B: Diretamente na Máquina Local (Desenvolvimento Rápido)
Use esta opção se preferir evitar o Docker localmente ou estiver com bloqueios de firewall.
```bash
.venv\Scripts\python api_scoring.py
```

*A API estará ativa em `http://localhost:8080/` e responderá na rota `/predict`.*

---

### 3. Executar o Simulador Comercial (Dashboard)
Visto que os navegadores modernos bloqueiam conexões AJAX (`fetch`) a partir do protocolo de ficheiro local `file:///`, servimos o dashboard de forma limpa:

```bash
# Inicie um servidor HTTP local simples usando o Python na pasta do projeto
.venv\Scripts\python -m http.server 8000
```

Agora, abra o seu browser favorito e aceda a:
👉 **[http://localhost:8000/dashboard_comercial.html](http://localhost:8000/dashboard_comercial.html)**

---

### 4. Executar o Pipeline de Retreino (Prefect)
Para simular um retreino mensal do modelo com validação automatizada de performance, execute:
```bash
.venv\Scripts\python pipeline_retreino.py
```
O Prefect gerará logs detalhados e assegurará a estabilidade do fluxo de dados antes de autorizar o deploy.

---

## 📊 Segmentação Comercial (Tiers de Leads)

A plataforma segmenta automaticamente cada predição em quatro categorias táticas:

| Tier | Limiar de Score | Ação Comercial Recomendada |
| :--- | :--- | :--- |
| 🔴 **Crítico** | `>= 80%` | Ligar HOJE — alto potencial de fecho imediato. |
| 🟡 **Quente** | `60% - 79%` | Agendar reunião comercial ainda esta semana. |
| 🔵 **Morno** | `30% - 59%` | Nutrir com conteúdo focado + follow-up em 15 dias. |
| ⚪ **Frio** | `< 30%` | Manter 100% sob automação de marketing digital. |

---

## 🔍 Revisões de Segurança e Resiliência Aplicadas
*   **Segurança de CORS:** A API foi configurada com `allow_credentials=False` permitindo o wildcard `allow_origins=["*"]`. Isto impede o bloqueio de requisições por parte de navegadores modernos.
*   **Compilação de Sistema:** O Dockerfile foi atualizado para instalar `libgomp1` de forma a garantir que o LightGBM realize a álgebra linear dentro do Linux de forma estável.
*   **Resolução de Nomes Local:** Dashboard alterado para realizar chamadas no domínio `localhost` em vez de IPs estáticos para total conformidade de resolução IPv4/IPv6 no Windows.
