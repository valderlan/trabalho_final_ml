# Trabalho Final - Machine Learning

##  Sumário

- [Descrição do Problema](#-descrição-do-problema)
- [Solução Desenvolvida](#-solução-desenvolvida)
- [Visão Geral](#-visão-geral)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Como Executar](#-como-executar)
  - [Opção 1: Executar Pipeline Completo](#opção-1-executar-pipeline-completo)
  - [Opção 2: Carregar Modelos Treinados](#opção-2-carregar-modelos-treinados)
- [Pipeline Completo](#-pipeline-completo)
  - [Etapa 1: Balanceamento do Dataset](#etapa-1-balanceamento-do-dataset)
  - [Etapa 2: Análise Exploratória (EDA)](#etapa-2-análise-exploratória-eda)
  - [Etapa 3: Pré-processamento](#etapa-3-pré-processamento)
  - [Etapa 4: Treinamento dos Modelos](#etapa-4-treinamento-dos-modelos)
  - [Etapa 5: Servindo a API](#etapa-5-servindo-a-api)
- [Uso da API](#-uso-da-api)
- [Artefatos Gerados](#-artefatos-gerados)

---

## Descrição do Problema

Este projeto aborda o desafio de **classificação de dados** utilizando Machine Learning, com foco em:

### Problema Principal:
Desenvolver um sistema completo de classificação que seja capaz de:
- Lidar com **dados desbalanceados** (classes com quantidades diferentes de amostras)
- **Comparar múltiplos algoritmos** de ML para identificar o mais adequado
- **Implantar em produção** de forma eficiente e escalável
- Fornecer **interpretabilidade** dos resultados e métricas de performance

### Desafios Técnicos:
1. **Balanceamento de grandes datasets**: Processar arquivos CSV grandes sem sobrecarregar a memória
2. **Limpeza e preparação**: Identificar e remover outliers, valores ausentes e features irrelevantes
3. **Seleção de modelo**: Avaliar comparativamente diferentes algoritmos (clássicos e deep learning)
4. **Otimização**: Grid Search para encontrar os melhores hiperparâmetros de cada modelo
5. **Deployment**: Exportar para formato universal e servir via API REST

---

## Solução Desenvolvida

A solução implementa um **pipeline end-to-end automatizado** que resolve todos os desafios:

### Componentes Principais:

#### 1.  **Balanceamento Inteligente** (`balancear_dataset.py`)
- Undersampling em streaming (processa em chunks de 200k linhas)
- Equilibra todas as classes ao tamanho da minoritária
- Eficiente em memória para grandes volumes de dados

#### 2.  **Análise Exploratória Automatizada** (`EDA.py`)
- Remoção automática de outliers usando método IQR
- Limpeza de valores ausentes (NaN)
- Geração de visualizações (distribuição de classes, matriz de correlação)
- Relatório completo em JSON com todas as estatísticas

#### 3.  **Pré-processamento Robusto** (`preprocessing.py`)
- Normalização com StandardScaler (média=0, desvio=1)
- Codificação de labels categóricas
- Divisão estratificada treino/teste (80/20)
- Persistência de transformadores para uso em produção

#### 4.  **Treinamento Comparativo** (`train.py`)
- **7 modelos** treinados simultaneamente:
  - Logistic Regression, KNN, Decision Tree
  - Random Forest, Extra Trees, MLP (Sklearn)
  - CNN 1D (PyTorch)
- **Grid Search** com validação cruzada (5-fold)
- **6 métricas** de avaliação: Accuracy, F1-Score, Recall, Precision, MCC, Log Loss
- Exportação universal para **ONNX**
- Seleção automática do melhor modelo (maior F1-Score)

#### 5.  **API REST Completa** (`app.py`)
- Framework **FastAPI** moderno e assíncrono
- **Dashboard HTML interativo** integrado:
  - Comparação visual de todos os modelos
  - Estatísticas do dataset (EDA)
  - Métricas de performance em tempo real
- **Inferência ONNX** otimizada
- **Seleção dinâmica** de modelo via API
- Documentação automática (Swagger UI + ReDoc)

### Resultados Obtidos:
-  Pipeline totalmente automatizado e reproduzível
-  7 modelos treinados, otimizados e comparados
-  Melhor modelo selecionado automaticamente
-  API REST pronta para produção
-  Dashboard para monitoramento e análise
-  Relatórios completos (JSON, CSV, gráficos)

---

##  Visão Geral

Este projeto implementa um **pipeline completo de Machine Learning** para classificação, desde o balanceamento de dados até a disponibilização de uma API REST para inferência. O sistema:

1. **Balanceia** datasets desbalanceados usando undersampling
2. **Analisa** os dados através de análise exploratória (EDA)
3. **Pré-processa** e normaliza as features
4. **Treina e compara** múltiplos modelos (Sklearn e PyTorch)
5. **Exporta** todos os modelos para ONNX
6. **Serve** predições através de uma API FastAPI com **Dashboard interativo**

###  Características da API

-  **Dashboard HTML**: Interface visual completa com métricas e estatísticas
-  **Multi-modelo**: Carregamento automático de todos os modelos ONNX
-  **Seleção dinâmica**: Escolha qual modelo usar em cada predição
-  **Métricas em tempo real**: Visualize performance de todos os modelos
-  **Comparação**: Compare predições de diferentes modelos
-  **Estatísticas EDA**: Dados da análise exploratória integrados ao dashboard

---

##  Requisitos

- **Python**: 3.13+
- **Dependências**: Definidas em `pyproject.toml`

### Principais bibliotecas:
- `scikit-learn`: Modelos clássicos de ML
- `torch`: Redes neurais (CNN)
- `pandas`, `numpy`: Manipulação de dados
- `matplotlib`, `seaborn`: Visualizações
- `fastapi`, `uvicorn`: API REST
- `onnx`, `onnxruntime`: Exportação e inferência de modelos
- `skl2onnx`: Conversão de modelos Sklearn para ONNX

---

##  Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/valderlan/trabalho_final_ml.git
cd trabalho_final_ml
```

### 2. Instale as dependências

**Usando uv (recomendado):**
```bash
uv sync
```

**Ou usando pip:**
```bash
pip install -e .
```

---

##  Como Executar

Existem duas formas de usar este projeto:

### Opção 1: Executar Pipeline Completo

Para treinar os modelos do zero com seus próprios dados:

```bash
# Passo 1: Balancear o dataset
python balancear_dataset.py

# Passo 2: Análise exploratória e limpeza
python EDA.py

# Passo 3: Pré-processar dados
python preprocessing.py

# Passo 4: Treinar todos os modelos
python train.py

# Passo 5: Iniciar a API
python app.py
```

**Observação:** Edite os scripts para apontar para seu arquivo CSV de entrada.

### Opção 2: Carregar Modelos Treinados

Se você já possui os modelos treinados (arquivos `.onnx`, `scaler.pkl`, etc.) na pasta `models/`:

```bash
# Inicia a API diretamente
python app.py
```

**Acesse:**
-  **Dashboard**: http://localhost:8000/
-  **API Docs (Swagger)**: http://localhost:8000/docs
-  **ReDoc**: http://localhost:8000/redoc

### Executar com Uvicorn (alternativa):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

O parâmetro `--reload` permite hot-reload durante desenvolvimento.

---

##  Estrutura do Projeto

Organização das pastas e arquivos do repositório:

```
trabalho_final_ml/
├── app.py                          # API FastAPI com dashboard
├── balancear_dataset.py            # Script de balanceamento
├── EDA.py                          # Análise exploratória
├── preprocessing.py                # Pré-processamento
├── train.py                        # Treinamento de modelos
├── pyproject.toml                  # Dependências do projeto
├── README.md                       # Esta documentação
├── relatorio_comparativo.md        # Relatório de comparação
│
├── dataset/                        # Dados
│   ├── dataset_balanceado.csv      # Dataset após balanceamento
│   ├── dataset_processed_final.csv # Dataset limpo (pós-EDA)
│   └── processed/                  # Arrays NumPy processados
│       ├── X_train.npy             # Features de treino
│       ├── X_test.npy              # Features de teste
│       ├── y_train.npy             # Labels de treino
│       ├── y_test.npy              # Labels de teste
│       └── model_comparison_results.csv # Comparação de modelos
│
├── models/                         # Modelos treinados
│   ├── best_model.onnx             # Melhor modelo (selecionado)
│   ├── scaler.pkl                  # Normalizador (StandardScaler)
│   ├── label_encoder.pkl           # Codificador de labels
│   ├── LogisticRegression.onnx     # Modelo: Regressão Logística
│   ├── KNN.onnx                    # Modelo: K-Nearest Neighbors
│   ├── DecisionTree.onnx           # Modelo: Árvore de Decisão
│   ├── RandomForest.onnx           # Modelo: Random Forest
│   ├── ExtraTrees.onnx             # Modelo: Extra Trees
│   ├── MLP_Sklearn.onnx            # Modelo: MLP (Sklearn)
│   └── CNN_PyTorch.onnx            # Modelo: CNN (PyTorch)
│
├── figures/                        # Visualizações
│   ├── class_distribution.png      # Distribuição de classes
│   ├── correlation_matrix.png      # Matriz de correlação
│   ├── cm_*.png                    # Matrizes de confusão
│   └── feature_importance_*.png    # Importância das features
│
├── json/                           # Metadados e relatórios
│   ├── metadata.json               # Configurações (n_features, classes)
│   ├── eda_report.json             # Relatório da EDA
│   └── training_metrics.json      # Métricas de todos os modelos
│
└── logs/                           # Logs de execução
    ├── eda_report.log              # Log da análise exploratória
    ├── preprocessing.log           # Log do pré-processamento
    └── training.log                # Log do treinamento
```

### Descrição das Pastas:

- **`dataset/`**: Armazena datasets em diferentes estágios (original, balanceado, processado)
- **`models/`**: Contém todos os modelos exportados em ONNX + artefatos (scaler, encoder)
- **`figures/`**: Visualizações geradas automaticamente (gráficos, matrizes de confusão)
- **`json/`**: Metadados e relatórios estruturados em JSON
- **`logs/`**: Logs detalhados de cada etapa do pipeline

---

##  Pipeline Completo

### Etapa 1: Balanceamento do Dataset

**Arquivo:** `balancear_dataset.py`

**Objetivo:** Equilibrar as classes do dataset através de undersampling, garantindo que todas as classes tenham o mesmo número de amostras da classe minoritária.

#### Como funciona:

1. **Leitura em chunks**: Processa grandes arquivos CSV em blocos de 200.000 linhas
2. **Contagem de classes**: Identifica a distribuição original de cada classe
3. **Undersampling**: Reduz as classes majoritárias ao tamanho da classe minoritária
4. **Escrita streaming**: Salva o dataset balanceado sem sobrecarregar a memória

#### Execução:

```bash
python balancear_dataset.py
```

**Parâmetros configuráveis no código:**
```python
input_csv = "dataset/dataset_original.csv"  # Dataset de entrada
target_column = "label"                      # Nome da coluna de classe
output_csv = "dataset/dataset_balanceado.csv" # Dataset de saída
```

**Saída:**
- `dataset/dataset_balanceado.csv`: Dataset com classes equilibradas

---

### Etapa 2: Análise Exploratória (EDA)

**Arquivo:** `EDA.py`

**Objetivo:** Realizar análise exploratória dos dados, identificar padrões, remover outliers e gerar visualizações.

#### Funcionalidades:

1. **Remoção de valores ausentes**:
   - Colunas: Remove colunas com qualquer valor NaN
   - Linhas: Opcionalmente remove linhas com NaN

2. **Remoção de outliers**:
   - Método IQR (Interquartile Range)
   - Identifica valores fora de Q1 - 1.5×IQR e Q3 + 1.5×IQR

3. **Geração de visualizações**:
   - Distribuição de classes
   - Matriz de correlação entre features
   - Boxplots e histogramas

4. **Relatório JSON**:
   - Estatísticas descritivas
   - Informações sobre limpeza
   - Análise de classes

#### Execução:

```bash
python EDA.py
```

**Parâmetros principais (modificar no código):**
```python
csv_path = "dataset/dataset_balanceado.csv"
label_column = "label"
remove_outliers = True   # Ativar/desativar remoção de outliers
drop_rows_with_nan = False
drop_cols_with_nan = True
```

**Saídas:**
- `dataset/dataset_processed_final.csv`: Dataset limpo
- `figures/class_distribution.png`: Gráfico de distribuição
- `figures/correlation_matrix.png`: Matriz de correlação
- `json/eda_report.json`: Relatório completo da análise
- `logs/eda_report.log`: Log detalhado do processo

---

### Etapa 3: Pré-processamento

**Arquivo:** `preprocessing.py`

**Objetivo:** Preparar os dados para treinamento através de normalização e divisão em conjuntos de treino/teste.

#### Processo:

1. **Carregamento**: Lê o dataset processado
2. **Separação**: Divide features (X) e target (y)
3. **Codificação**: Converte labels categóricas em numéricas usando `LabelEncoder`
4. **Divisão**: Split 80/20 para treino/teste com estratificação
5. **Normalização**: Aplica `StandardScaler` (média=0, desvio=1)
6. **Salvamento**: Persiste os arrays NumPy e transformadores

#### Execução:

```bash
python preprocessing.py
```

**Parâmetros (modificar no código):**
```python
input_csv = "dataset/dataset_processed_final.csv"
target_col = "label"
test_size = 0.2  # 20% para teste
```

**Saídas:**
- `dataset/processed/X_train.npy`: Features de treino normalizadas
- `dataset/processed/X_test.npy`: Features de teste normalizadas
- `dataset/processed/y_train.npy`: Labels de treino codificadas
- `dataset/processed/y_test.npy`: Labels de teste codificadas
- `models/scaler.pkl`: StandardScaler treinado
- `models/label_encoder.pkl`: LabelEncoder para decodificação
- `json/metadata.json`: Metadados (n_features, n_classes, mapeamento)
- `logs/preprocessing.log`: Log do processo

---

### Etapa 4: Treinamento dos Modelos

**Arquivo:** `train.py`

**Objetivo:** Treinar múltiplos modelos de classificação, comparar desempenho e exportar o melhor para ONNX.

#### Modelos Treinados:

**Sklearn:**
1. **Logistic Regression**: Classificação linear
2. **K-Nearest Neighbors (KNN)**: Baseado em proximidade
3. **Decision Tree**: Árvore de decisão
4. **Random Forest**: Ensemble de árvores
5. **Extra Trees**: Ensemble com divisões aleatórias
6. **MLP (Multi-Layer Perceptron)**: Rede neural Sklearn

**PyTorch:**
7. **CNN (Convolutional Neural Network)**: Rede convolucional 1D

#### Grid Search:

Cada modelo passa por **Grid Search com validação cruzada** (5-fold) para encontrar os melhores hiperparâmetros.

Exemplos de grids:
```python
"LogisticRegression": {"C": [0.1, 1, 10]}
"KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
"RandomForest": {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
```

#### Métricas Calculadas:

- **Accuracy**: Acurácia geral
- **F1-Score (Macro)**: Média harmônica de precisão e recall
- **Recall (Macro)**: Taxa de verdadeiros positivos
- **Precision (Macro)**: Taxa de predições corretas
- **MCC (Matthews Correlation Coefficient)**: Correlação entre predito e real
- **Log Loss**: Perda logarítmica (quando aplicável)

#### Visualizações:

Para cada modelo:
- **Matriz de Confusão**: `figures/cm_{modelo}.png`
- **Feature Importance** (quando disponível): `figures/feature_importance_{modelo}.png`

#### Exportação ONNX:

Todos os modelos são exportados para o formato **ONNX** (Open Neural Network Exchange):
- Permite inferência rápida e independente de framework
- Compatível com múltiplas linguagens e plataformas

#### Execução:

```bash
python train.py
```

**Processo automático:**
1. Carrega dados do pré-processamento
2. Para cada modelo:
   - Executa Grid Search
   - Treina com melhores parâmetros
   - Avalia em treino e teste
   - Gera visualizações
   - Exporta para ONNX
3. Compara todos os modelos
4. Copia o melhor para `models/best_model.onnx`

**Saídas:**
- `models/{modelo}.onnx`: Cada modelo em ONNX
- `models/best_model.onnx`: Melhor modelo (cópia)
- `dataset/processed/model_comparison_results.csv`: Comparação de métricas
- `figures/cm_*.png`: Matrizes de confusão
- `figures/feature_importance_*.png`: Importância de features
- `json/training_metrics.json`: Métricas detalhadas de todos os modelos
- `logs/training.log`: Log completo do treinamento

#### Critério de Seleção do Melhor Modelo:

O modelo com **maior F1-Score Macro no conjunto de teste** é escolhido como o melhor.

---

### Etapa 5: Servindo a API

**Arquivo:** `app.py`

**Objetivo:** Disponibilizar uma API REST completa com dashboard interativo e predições usando múltiplos modelos ONNX.

#### Arquitetura:

- **Framework**: FastAPI (moderna, rápida, assíncrona)
- **Inferência**: ONNX Runtime (otimizado para produção)
- **Servidor**: Uvicorn (ASGI server)
- **Dashboard**: Interface web HTML integrada

#### Inicialização (Lifespan):

No startup, a API carrega automaticamente:

1. **Metadados e Relatórios:**
   - `json/metadata.json`: Configurações dos modelos (features, classes)
   - `json/eda_report.json`: Estatísticas da análise exploratória
   - `json/training_metrics.json`: Métricas de performance dos modelos

2. **Artefatos de ML:**
   - `models/scaler.pkl`: StandardScaler para normalização
   - `models/*.onnx`: **TODOS** os modelos ONNX disponíveis (não apenas o best_model)

3. **Carregamento Dinâmico:**
   - A API detecta e carrega automaticamente todos os arquivos `.onnx` na pasta `models/`
   - Cada modelo fica disponível para inferência individual

#### Endpoints Disponíveis:

##### 1. **GET** `/` - Dashboard Interativo

Página principal com dashboard HTML contendo:

-  **Status da API**: Indicador de saúde e modelos carregados
-  **Tabela de Performance**: Comparação de todos os modelos com métricas
  - Accuracy, F1-Score, Recall, Precision, MCC, Log Loss
  - Overfitting Gap (diferença entre treino e teste)
  - Destaque visual do melhor modelo
-  **Estatísticas do Dataset**:
  - Total de amostras e features
  - Colunas removidas durante EDA
  - Distribuição de classes
-  **Detalhes Expandíveis**:
  - Estatísticas descritivas de cada feature (média, desvio, min, max, mediana)
  - Análise de perda por classe durante limpeza
-  **Link direto** para `/docs` (Swagger)

**Acesso:** `http://localhost:8000/`

##### 2. **POST** `/predict` - Inferência com Seleção de Modelo

Realiza predições usando o modelo especificado (padrão: `best_model`).

**Parâmetros:**
```json
{
  "features": [0.5, 1.2, -0.3, ...],
  "model_name": "RandomForest"  // Opcional, padrão = "best_model"
}
```

**Modelos disponíveis para seleção:**
- `best_model` (padrão)
- `LogisticRegression`
- `KNN`
- `DecisionTree`
- `RandomForest`
- `ExtraTrees`
- `MLP_Sklearn`
- `CNN_PyTorch`

#### Execução:

```bash
python app.py
```

Ou diretamente com uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Servidor rodando em:** `http://localhost:8000`

**Recursos disponíveis:**
-  Dashboard: `http://localhost:8000/`
-  Swagger UI: `http://localhost:8000/docs`
-  ReDoc: `http://localhost:8000/redoc`

#### Principais Funcionalidades da API:


1. **Dashboard HTML Completo** (`GET /`):
   - Interface visual rica e responsiva
   - Tabela comparativa com todos os modelos e suas métricas
   - Destaque automático do melhor modelo (fundo verde)
   - Estatísticas do dataset (amostras, features, classes)
   - Estatísticas descritivas detalhadas (média, desvio, mediana, etc.)
   - Distribuição de classes com análise de perda
   - Seções expandíveis para detalhes adicionais
   - Link direto para `/docs`

2. **Carregamento Multi-Modelo Automático**:
   - Detecta e carrega **todos** os modelos `.onnx` da pasta `models/`
   - Não apenas o `best_model`, mas todos os modelos treinados
   - Possibilita comparação em tempo real

3. **Seleção Dinâmica de Modelo** (`POST /predict`):
   - Parâmetro opcional `model_name` para escolher o modelo
   - Se omitido, usa `best_model` como padrão
   - Resposta inclui qual modelo foi utilizado

4. **Integração com Relatórios JSON**:
   - Carrega automaticamente `eda_report.json`, `metadata.json` e `training_metrics.json`
   - Exibe métricas de performance de todos os modelos
   - Mostra estatísticas da análise exploratória
   - Apresenta overfitting gap (diferença treino vs teste)

5. **Validação Robusta**:
   - Verifica existência do modelo solicitado
   - Valida número de features
   - Mensagens de erro descritivas com modelos disponíveis

---

##  Uso da API

### 1. Acessar o Dashboard

Abra o navegador e acesse: `http://localhost:8000/`

Você verá uma interface completa com:
- Status da API e modelos carregados
- Comparação de performance de todos os modelos
- Estatísticas detalhadas do dataset
- Destaque do melhor modelo

### 2. Fazer uma predição

#### Opção A: Via Dashboard (Swagger UI)

1. Acesse: `http://localhost:8000/docs`
2. Clique em **POST /predict**
3. Clique em **Try it out**
4. Insira o JSON de entrada
5. Clique em **Execute**

#### Opção B: Via cURL (usando modelo padrão)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1]
  }'
```

#### Opção C: Via cURL (selecionando modelo específico)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1],
    "model_name": "RandomForest"
  }'
```

#### Opção D: Via Python (modelo padrão)

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1]
}

response = requests.post(url, json=data)
print(response.json())
```

#### Opção E: Via Python (selecionando modelo)

```python
import requests

url = "http://localhost:8000/predict"

# Testando com diferentes modelos
modelos = ["best_model", "RandomForest", "KNN", "CNN_PyTorch"]

for modelo in modelos:
    data = {
        "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1],
        "model_name": modelo
    }
    
    response = requests.post(url, json=data)
    resultado = response.json()
    print(f"{modelo}: {resultado['class_name']} (ID: {resultado['prediction_id']})")
```

#### Resposta esperada:

```json
{
  "prediction_id": 1,
  "class_name": "ClasseA",
  "model_used": "RandomForest",
  "model_type": "ONNX Inference"
}
```

### 3. Validações Automáticas:

A API valida automaticamente:
-  Número correto de features
-  Tipos de dados válidos
-  Modelo especificado existe
-  Artefatos carregados corretamente

#### Exemplos de Erros:

**Erro 1: Número de features incorreto**
```json
{
  "detail": "Esperado 10 features."
}
```

**Erro 2: Modelo não encontrado**
```json
{
  "detail": "Modelo 'SVM' não encontrado. Opções disponíveis: ['best_model', 'LogisticRegression', 'KNN', 'DecisionTree', 'RandomForest', 'ExtraTrees', 'MLP_Sklearn', 'CNN_PyTorch']"
}
```

### 4. Comparação de Modelos

Use o endpoint `/predict` com diferentes valores de `model_name` para comparar as predições de múltiplos modelos no mesmo input:

```python
import requests

features = [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1]

modelos = ["best_model", "RandomForest", "KNN", "LogisticRegression"]
resultados = {}

for modelo in modelos:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": features, "model_name": modelo}
    )
    resultados[modelo] = response.json()

# Exibir comparação
for modelo, resultado in resultados.items():
    print(f"{modelo:20} -> {resultado['class_name']}")
```

### 5. Visualizar Métricas no Dashboard

O dashboard HTML (`http://localhost:8000/`) apresenta informações detalhadas:

#### Seção 1: Status e Modelos Carregados
- Badge de status (ONLINE)
- Total de modelos carregados
- Nome do melhor modelo destacado
- Tags com todos os modelos disponíveis

#### Seção 2: Performance dos Modelos
Tabela comparativa ordenada por F1-Score com:
- **Accuracy**: Acurácia geral
- **F1-Macro**: Média harmônica (métrica principal)
- **Recall**: Taxa de verdadeiros positivos
- **Precision**: Taxa de acerto das predições
- **MCC**: Correlação de Matthews
- **Log Loss**: Perda logarítmica (quando aplicável)
- **Overfitting Gap**: Diferença entre métricas de treino e teste
- **Destaque verde** no melhor modelo

#### Seção 3: Estatísticas do Dataset
- Cards com totais: amostras, features, colunas removidas
- Colunas ignoradas durante EDA (em vermelho)
- **Estatísticas Descritivas** (expandível):
  - Tabela com média, desvio padrão, mínimo, mediana, máximo de cada feature
- **Distribuição de Classes** (expandível):
  - Quantidade final de cada classe
  - Perda durante limpeza (outliers, NaN)


---

##  Artefatos Gerados

### Datasets:
- `dataset_balanceado.csv`: Classes equilibradas
- `dataset_processed_final.csv`: Dados limpos e prontos
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`: Arrays normalizados

### Modelos:
- `*.onnx`: Modelos exportados
- `best_model.onnx`: Modelo de produção
- `scaler.pkl`, `label_encoder.pkl`: Transformadores

### Relatórios:
- `eda_report.json`: Estatísticas e análise
- `metadata.json`: Configurações e mapeamentos
- `training_metrics.json`: Métricas de todos os modelos
- `model_comparison_results.csv`: Comparação tabular

### Visualizações:
- `class_distribution.png`: Distribuição de classes
- `correlation_matrix.png`: Correlações entre features
- `cm_*.png`: Matrizes de confusão de cada modelo
- `feature_importance_*.png`: Importância de features

### Logs:
- `eda_report.log`: Log da análise exploratória
- `preprocessing.log`: Log do pré-processamento
- `training.log`: Log detalhado do treinamento

## Resumo do Projeto

### Tecnologias Utilizadas:
- **Python 3.13+**
- **Scikit-learn**: Modelos clássicos de ML
- **PyTorch**: Deep Learning (CNN)
- **FastAPI**: API REST moderna
- **ONNX**: Formato universal de modelos
- **Pandas/NumPy**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualizações

### Modelos Implementados:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Extra Trees
6. Multi-Layer Perceptron (MLP)
7. Convolutional Neural Network (CNN)

### Métricas de Avaliação:
- Accuracy
- F1-Score (Macro)
- Recall (Macro)
- Precision (Macro)
- Matthews Correlation Coefficient (MCC)
- Log Loss




