# Lead Scoring B2B — Machine Learning para Priorização de Leads Comerciais

Projeto de Ciência de Dados end-to-end para priorização automática de leads B2B, utilizando Machine Learning para estimar a probabilidade de conversão de cada oportunidade comercial.

O objetivo do projeto é apoiar times de vendas na tomada de decisão, ajudando a identificar quais leads devem receber maior prioridade de contato com base em dados históricos, características do lead e padrões de conversão.

---

## Visão Geral

Em operações comerciais B2B, nem todos os leads possuem o mesmo potencial de conversão. Quando não há uma priorização baseada em dados, o time comercial pode gastar tempo com oportunidades de baixa probabilidade enquanto leads com maior potencial deixam de ser trabalhados no momento certo.

Este projeto propõe uma solução de **Lead Scoring baseada em Machine Learning**, capaz de classificar leads conforme sua probabilidade de conversão e apoiar a priorização comercial.

A solução contempla:

* Análise exploratória dos dados;
* Tratamento e preparação das variáveis;
* Treinamento de modelo preditivo;
* Avaliação de performance;
* Geração de score de conversão;
* Estruturação de pipeline de inferência;
* Disponibilização do modelo por API;
* Containerização com Docker.

---

## Problema de Negócio

O time comercial precisa responder a uma pergunta central:

> Quais leads têm maior probabilidade de conversão e devem ser priorizados pelo time de vendas?

Sem um modelo de priorização, a abordagem comercial tende a depender apenas de critérios manuais, percepção individual ou ordem de chegada dos leads. Isso pode reduzir a eficiência do processo de vendas e dificultar a alocação inteligente do esforço comercial.

---

## Objetivo do Projeto

Desenvolver uma solução de Machine Learning capaz de:

* Estimar a probabilidade de conversão de leads B2B;
* Classificar leads por nível de prioridade;
* Apoiar decisões comerciais com base em dados;
* Reduzir esforço manual na triagem de oportunidades;
* Criar uma estrutura reutilizável para treinamento e inferência do modelo.

---

## Abordagem Utilizada

O projeto foi estruturado seguindo as principais etapas de um fluxo de Ciência de Dados:

1. **Entendimento do problema de negócio**
   Definição do objetivo analítico e da métrica de sucesso.

2. **Análise exploratória de dados**
   Investigação das variáveis, distribuição dos dados, padrões de conversão e possíveis inconsistências.

3. **Preparação dos dados**
   Tratamento de valores ausentes, transformação de variáveis, encoding de categorias e estruturação da base para modelagem.

4. **Treinamento do modelo**
   Desenvolvimento de modelo supervisionado utilizando LightGBM para classificação dos leads.

5. **Avaliação de performance**
   Análise de métricas como AUC, precisão, recall e matriz de confusão para avaliar a capacidade preditiva do modelo.

6. **Geração de score**
   Conversão das probabilidades previstas em uma pontuação de priorização comercial.

7. **Inferência e deploy local**
   Estruturação de API com FastAPI e containerização com Docker para simular o uso prático do modelo.

---

## Tecnologias Utilizadas

### Linguagem e Bibliotecas

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM

### Machine Learning

* Classificação supervisionada
* Feature Engineering
* Validação de modelo
* Métricas de classificação
* Lead Scoring

### Deploy e MLOps

* FastAPI
* Docker
* Pipeline de treinamento
* Pipeline de inferência

### Visualização e Análise

* Matplotlib
* Seaborn
* Análise exploratória de dados
* Avaliação de métricas

---

## Resultado Obtido

O modelo desenvolvido com LightGBM alcançou **AUC aproximada de 0,83**, demonstrando boa capacidade de separação entre leads com maior e menor probabilidade de conversão.

Esse resultado indica que o modelo consegue apoiar a priorização comercial ao identificar padrões relevantes nos dados e transformar esses padrões em uma pontuação preditiva para cada lead.

---

## Estrutura do Projeto

```text
lead-scoring-b2b-mlops/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│
├── api/
│   ├── main.py
│
├── models/
│   ├── lead_scoring_model.pkl
│
├── reports/
│   ├── figures/
│   ├── metrics.json
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Principais Entregáveis

* Notebook de análise exploratória;
* Pipeline de tratamento e preparação dos dados;
* Modelo de classificação com LightGBM;
* Avaliação de performance do modelo;
* Geração de score de conversão;
* API para inferência do modelo;
* Estrutura Docker para execução da aplicação.

---

## Métricas Avaliadas

As principais métricas utilizadas para avaliação do modelo foram:

* **AUC ROC**: mede a capacidade do modelo de separar leads convertidos e não convertidos;
* **Precisão**: avalia a proporção de leads classificados como positivos que realmente converteram;
* **Recall**: avalia a capacidade do modelo de encontrar leads com potencial de conversão;
* **Matriz de confusão**: permite visualizar acertos e erros de classificação;
* **Distribuição dos scores**: analisa a segmentação dos leads por nível de prioridade.

---

## Exemplo de Uso da API

Após treinar o modelo e iniciar a API, é possível enviar as características de um lead e receber como resposta a probabilidade de conversão.

### Requisição

```json
{
  "company_size": "medium",
  "segment": "technology",
  "lead_source": "paid_media",
  "number_of_interactions": 5,
  "days_since_first_contact": 12,
  "has_requested_demo": 1
}
```

### Resposta

```json
{
  "conversion_probability": 0.78,
  "lead_score": 78,
  "priority": "high"
}
```

---

## Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/guilhermehrsilva/lead-scoring-b2b-mlops.git
cd lead-scoring-b2b-mlops
```

### 2. Criar ambiente virtual

```bash
python -m venv venv
```

### 3. Ativar ambiente virtual

No Windows:

```bash
venv\Scripts\activate
```

No Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

### 5. Executar treinamento do modelo

```bash
python src/train_model.py
```

### 6. Iniciar a API

```bash
uvicorn api.main:app --reload
```

A API ficará disponível em:

```text
http://127.0.0.1:8000
```

A documentação interativa poderá ser acessada em:

```text
http://127.0.0.1:8000/docs
```

---

## Execução com Docker

### 1. Construir a imagem

```bash
docker build -t lead-scoring-b2b .
```

### 2. Executar o container

```bash
docker run -p 8000:8000 lead-scoring-b2b
```

---

## Possíveis Aplicações de Negócio

Este tipo de solução pode ser aplicado em contextos como:

* Priorização de leads comerciais;
* Segmentação de oportunidades;
* Apoio à estratégia de vendas;
* Redução de esforço manual na triagem de leads;
* Otimização da abordagem comercial;
* Acompanhamento de performance por canal de aquisição;
* Identificação de leads com maior potencial de conversão.

---

## Aprendizados do Projeto

Este projeto reforça competências importantes para atuação em Ciência de Dados aplicada a negócios:

* Tradução de um problema comercial em um problema de classificação;
* Preparação e validação de dados para Machine Learning;
* Construção de modelo preditivo com LightGBM;
* Avaliação de métricas de classificação;
* Interpretação de resultados para tomada de decisão;
* Estruturação de pipeline de inferência;
* Disponibilização de modelo via API;
* Uso de Docker para padronização do ambiente.

---

## Próximas Melhorias

Algumas evoluções possíveis para o projeto:

* Adicionar explicabilidade do modelo com SHAP;
* Criar dashboard para monitoramento dos scores;
* Implementar monitoramento de drift dos dados;
* Adicionar testes automatizados;
* Criar pipeline completo de retreinamento;
* Comparar diferentes algoritmos de classificação;
* Simular impacto financeiro da priorização dos leads;
* Criar segmentação por faixas de score para diferentes estratégias comerciais.

---

## Autor

**Guilherme Henrique Risson Silva**

* LinkedIn: [linkedin.com/in/guilhermerisson](https://www.linkedin.com/in/guilhermerisson)
* GitHub: [github.com/guilhermehrsilva](https://github.com/guilhermehrsilva)

---

## Observação

Este projeto foi desenvolvido com foco em portfólio de Ciência de Dados, demonstrando a construção de uma solução prática de Machine Learning aplicada a um problema real de negócio: priorização comercial baseada em dados.
