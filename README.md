# Trabalho Final - Machine Learning

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Pipeline Completo](#-pipeline-completo)
  - [Etapa 1: Balanceamento do Dataset](#etapa-1-balanceamento-do-dataset)
  - [Etapa 2: Análise Exploratória (EDA)](#etapa-2-análise-exploratória-eda)
  - [Etapa 3: Pré-processamento](#etapa-3-pré-processamento)
  - [Etapa 4: Treinamento dos Modelos](#etapa-4-treinamento-dos-modelos)
  - [Etapa 5: Servindo a API](#etapa-5-servindo-a-api)
- [Uso da API](#-uso-da-api)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Artefatos Gerados](#-artefatos-gerados)

---

## 🎯 Visão Geral

Este projeto implementa um **pipeline completo de Machine Learning** para classificação, desde o balanceamento de dados até a disponibilização de uma API REST para inferência. O sistema:

1. **Balanceia** datasets desbalanceados usando undersampling
2. **Analisa** os dados através de análise exploratória (EDA)
3. **Pré-processa** e normaliza as features
4. **Treina e compara** múltiplos modelos (Sklearn e PyTorch)
5. **Exporta** o melhor modelo para ONNX
6. **Serve** predições através de uma API FastAPI

---

## 📦 Requisitos

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

## 🚀 Instalação

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

## 🔄 Pipeline Completo

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

**Objetivo:** Disponibilizar uma API REST para realizar predições usando o melhor modelo treinado.

#### Arquitetura:

- **Framework**: FastAPI (moderna, rápida, assíncrona)
- **Inferência**: ONNX Runtime (otimizado para produção)
- **Servidor**: Uvicorn (ASGI server)

#### Inicialização:

No startup, a API carrega:
1. `json/metadata.json`: Configurações do modelo
2. `models/scaler.pkl`: Para normalização dos inputs
3. `models/best_model.onnx`: Modelo para inferência

#### Endpoint Disponível:

**POST** `/predict`

Recebe um JSON com as features e retorna a predição.

#### Execução:

```bash
python app.py
```

Ou diretamente com uvicorn:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Servidor rodando em:** `http://localhost:8000`

**Documentação interativa:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🔌 Uso da API

### 1. Verificar se a API está rodando

Acesse no navegador: `http://localhost:8000/docs`

### 2. Fazer uma predição

#### Via cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1]
  }'
```

#### Via Python:

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": [0.5, 1.2, -0.3, 0.8, 1.5, 0.2, -0.7, 0.9, 0.4, -0.1]
}

response = requests.post(url, json=data)
print(response.json())
```

#### Resposta esperada:

```json
{
  "prediction_id": 1,
  "class_name": "ClasseA",
  "model_type": "ONNX Inference"
}
```

### 3. Validações:

A API valida automaticamente:
- ✅ Número correto de features
- ✅ Tipos de dados válidos
- ✅ Modelo carregado corretamente

#### Erro se número de features incorreto:

```json
{
  "detail": "Esperado 10 features."
}
```

---

## 📁 Estrutura do Projeto

```
trabalho_final_ml/
├── app.py                          # API FastAPI
├── balancear_dataset.py            # Balanceamento de classes
├── EDA.py                          # Análise exploratória
├── preprocessing.py                # Pré-processamento
├── train.py                        # Treinamento de modelos
├── pyproject.toml                  # Dependências e metadados
├── README.md                       # Esta documentação
├── relatorio_comparativo.md        # Comparação de modelos
│
├── dataset/                        # Dados
│   ├── dataset_balanceado.csv      # Dataset balanceado
│   ├── dataset_processed_final.csv # Dataset limpo (EDA)
│   └── processed/                  # Arrays NumPy
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── model_comparison_results.csv
│
├── models/                         # Modelos treinados
│   ├── best_model.onnx             # Melhor modelo
│   ├── scaler.pkl                  # StandardScaler
│   ├── label_encoder.pkl           # LabelEncoder
│   ├── LogisticRegression.onnx
│   ├── KNN.onnx
│   ├── DecisionTree.onnx
│   ├── RandomForest.onnx
│   ├── ExtraTrees.onnx
│   ├── MLP_Sklearn.onnx
│   └── CNN_PyTorch.onnx
│
├── figures/                        # Visualizações
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── cm_*.png                    # Matrizes de confusão
│   └── feature_importance_*.png
│
├── json/                           # Metadados e relatórios
│   ├── metadata.json               # Configurações do modelo
│   ├── eda_report.json             # Relatório da EDA
│   └── training_metrics.json      # Métricas de treinamento
│
└── logs/                           # Logs de execução
    ├── eda_report.log
    ├── preprocessing.log
    └── training.log
```

---

## 📊 Artefatos Gerados

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

---

## 🎓 Fluxo Completo de Execução

Para executar o pipeline completo do zero:

```bash
# 1. Balancear o dataset
python balancear_dataset.py

# 2. Realizar análise exploratória
python EDA.py

# 3. Pré-processar os dados
python preprocessing.py

# 4. Treinar todos os modelos
python train.py

# 5. Iniciar a API
python app.py
```

Após esses passos, você terá:
- ✅ Dataset balanceado e limpo
- ✅ Visualizações e relatórios
- ✅ 7 modelos treinados e comparados
- ✅ API REST servindo o melhor modelo

---

## 🔧 Customizações

### Adicionar novos modelos:

Edite `train.py` e adicione no dicionário `models_config`:

```python
"SeuModelo": {
    "model": SeuClassificador(),
    "params": {"param1": [val1, val2], "param2": [val3, val4]}
}
```

### Modificar hiperparâmetros:

Ajuste os grids de busca em `train.py`:

```python
"RandomForest": {
    "model": RandomForestClassifier(random_state=42),
    "params": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
}
```

### Alterar split de treino/teste:

Em `preprocessing.py`, modifique:

```python
test_size = 0.3  # 30% para teste
```

---

## 📝 Notas Importantes

1. **Memória**: O balanceamento em streaming permite processar datasets grandes sem problemas
2. **ONNX**: Formato universal para modelos, possibilita deploy em diversas plataformas
3. **Grid Search**: Pode ser demorado para modelos complexos; ajuste os grids conforme necessário
4. **Logs**: Sempre verifique os logs em caso de erros
5. **API**: Em produção, considere adicionar autenticação e rate limiting

---

## 🤝 Contribuições

Este projeto é parte do trabalho final da disciplina de Machine Learning e Mineração de Dados.

**Autor:** Valderlan  
**Repositório:** [github.com/valderlan/trabalho_final_ml](https://github.com/valderlan/trabalho_final_ml)

---

## 📄 Licença

[Especifique a licença aqui]