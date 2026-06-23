# projeto-3w-aws

**Detecção e Classificação de Eventos Indesejáveis em Poços Offshore com Machine Learning e Engenharia de Dados**

Trabalho de Conclusão de Curso (TCC) — Especialização em Engenharia de Dados e Big Data  
USP Escola Politécnica — Programa eEDB-007  
Autor: Alan Pinheiro da Silva | 2026

---

<!-- Language & Platform -->
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-AWS-7B42BC?style=flat&logo=terraform&logoColor=white)

<!-- Data Engineering -->
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9.2-017CEE?style=flat&logo=apacheairflow&logoColor=white)
![AWS Glue](https://img.shields.io/badge/AWS%20Glue-4.0-FF9900?style=flat&logo=amazonaws&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-ETL-E25A1C?style=flat&logo=apachespark&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-S3%20Compatible-C72E49?style=flat&logo=minio&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=flat&logo=postgresql&logoColor=white)

<!-- Machine Learning -->
![LightGBM](https://img.shields.io/badge/LightGBM-F1%2096.2%25-brightgreen?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-F1%2095.7%25-green?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RF%2091.1%25-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-blueviolet?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blue?style=flat)

<!-- Domain -->
![Dataset](https://img.shields.io/badge/Dataset-Petrobras%203W-black?style=flat)
![Domain](https://img.shields.io/badge/Domain-Offshore%20Oil%20Wells-0077B6?style=flat)
![Architecture](https://img.shields.io/badge/Architecture-Medallion%20Bronze%2FSilver%2FGold-gold?style=flat)

---

## Visão Geral

Este projeto desenvolve um sistema automatizado de detecção e classificação de eventos indesejáveis em poços de petróleo offshore, utilizando o dataset público [3W da Petrobras](https://github.com/petrobras/3W) como base de dados. O pipeline processa séries temporais de 8 sensores de processo e classifica 6 classes de eventos com F1-macro de **96,2%** (LightGBM).

**Escopo de dados:** poços WELL-00002, WELL-00004 e WELL-00006 — 7,5 milhões de registros, 6 classes de eventos.

---

## Resultados dos Modelos

| Modelo | F1-Macro Teste | Acurácia Teste | Observação |
|--------|---------------|----------------|------------|
| Random Forest | 91,1% | 99,6% | 150 árvores, class_weight=balanced |
| XGBoost | 90,0% → 95,7%* | 99,3% | *após threshold tuning (+5,7pp) |
| **LightGBM** ★ | **96,2%** | **99,9%** | Melhor modelo — recomendado para produção |

Split estratificado por classe (70/15/15). O split temporal foi descartado pois excluía a classe 7 do treino (F1-macro RF caiu para 24,5%).

---

## Arquitetura do Pipeline

```
GitHub 3W API
     │
     ▼
[INGESTÃO]  ingestao_unificada.py  →  data/raw/
     │
     ▼
[BRONZE]    schema padronizado + campo bronze_at  →  data/bronze/
     │
     ▼
[SILVER]    validação física + fill NaN + dedup  →  data/silver/
     │
     ▼
[GOLD]      136 features + Z-Score + split  →  data/gold/{train,val,test}.parquet
     │
     ▼
[ML]        RF + XGBoost + LightGBM → Threshold Tuning → SHAP
```

**Features Gold (136 total):**  
Rolling statistics (média, std, min, max × janelas 5/10/30 × 8 sensores = 96) + Lag features (passos 1/3/5 × 8 = 24) + Delta (8) + Sensores brutos (8)

---

## Stack de Infraestrutura

| Camada | Tecnologia |
|--------|-----------|
| Processamento ETL | AWS Glue 4.0 (container Docker local) + PySpark |
| Orquestração | Apache Airflow 2.9.2 (LocalExecutor + PostgreSQL) |
| Armazenamento objetos | MinIO (emulador S3 local) |
| Banco de metadados | PostgreSQL 15 |
| Notebooks / EDA | Jupyter (scipy-notebook) |
| ML | scikit-learn, XGBoost, LightGBM, SHAP, Optuna |
| Infraestrutura nuvem | Terraform (AWS — deploy pendente) |

Toda a stack local é orquestrada por `docker-compose.yml`.

---

## Estrutura do Repositório

```
projeto-3w-aws/
├── config/
│   └── spark_config.py          # Caminhos, credenciais via env var, SparkSession lazy
├── dags/
│   ├── dag_etl_3w_pipeline.py   # ETL Bronze→Silver→Gold
│   ├── dag_gold_rebuild.py      # Re-processa apenas a camada Gold
│   └── dag_ml_training.py       # RF+XGB+LGB paralelo → Tuning → SHAP
├── data/
│   ├── raw/                     # Parquet por classe (gitignored)
│   ├── bronze/                  # Schema padronizado (gitignored)
│   ├── silver/                  # Dados validados (gitignored)
│   └── gold/                    # Features + splits (gitignored)
├── docker/
│   └── init_db.sql              # Schema PostgreSQL
├── models/                      # Métricas JSON + gráficos (joblib gitignored)
├── notebooks/
│   ├── eda_3w.ipynb             # Análise Exploratória — Figuras 1–9
│   └── dashboard_3w.ipynb       # Dashboard interativo
├── scripts/
│   ├── ingestao_unificada.py    # Download GitHub API (substitui 10 scripts por classe)
│   ├── etl_bronze_silver_gold.py
│   ├── train_rf_baseline.py
│   ├── train_xgb_baseline.py
│   ├── train_lgb_baseline.py
│   ├── tune_thresholds.py
│   └── shap_analysis.py
├── terraform/                   # Infraestrutura AWS (deploy pendente)
├── .env.example                 # Template de variáveis de ambiente
├── .gitignore
├── docker-compose.yml
└── requirements.txt
```

---

## Como Executar

### 1. Pré-requisitos

- Docker Desktop instalado e em execução
- Python 3.9+
- Token GitHub (recomendado para evitar rate limit de 60 req/hora):  
  gere em https://github.com/settings/tokens com escopo `public_repo`

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env e defina GITHUB_TOKEN, senhas PostgreSQL etc.
```

### 3. Subir a stack Docker

```bash
docker compose up -d
# Airflow UI: http://localhost:8080  
# Jupyter:    http://localhost:8888
# MinIO:      http://localhost:9000  
# pgAdmin:    http://localhost:5050
```

### 4. Instalar dependências Python (fora do container)

```bash
pip install -r requirements.txt
```

### 5. Executar o pipeline

**Via Airflow (recomendado):**
```
Airflow UI → DAGs → dag_etl_3w_pipeline → Trigger DAG
Airflow UI → DAGs → dag_ml_training     → Trigger DAG
```

**Via scripts diretos:**
```bash
# Ingestão
python scripts/ingestao_unificada.py --classes 0 1 2 4 6 7 --all-wells

# ETL completo
python scripts/etl_bronze_silver_gold.py --step all

# Treinamento
python scripts/train_lgb_baseline.py
python scripts/tune_thresholds.py
python scripts/shap_analysis.py
```

---

## Segurança

- `GITHUB_TOKEN`, senhas PostgreSQL e chaves MinIO: **sempre via variáveis de ambiente**, nunca hardcoded
- `.env` real está no `.gitignore` — use `.env.example` como template
- Modelos serializados (`*.joblib`) estão no `.gitignore` — use Git LFS se necessário versionar

---

## Classes de Eventos — Escopo do Projeto

| Classe | Evento | Presente nos 3 poços? |
|--------|--------|-----------------------|
| 0 | Normal | ✅ 334 arquivos |
| 1 | Abrupt Increase of BSW | ✅ 3 arquivos |
| 2 | Spurious Closure of DHSV | ✅ 1 arquivo |
| 3 | Severe Slugging | ❌ Ausente nos poços 00002/00004/00006 |
| 4 | Flow Instability | ✅ 155 arquivos |
| 5 | Rapid Productivity Loss | ❌ Ausente nos poços 00002/00004/00006 |
| 6 | Quick Restriction in PCK | ✅ 6 arquivos |
| 7 | Scaling in PCK | ✅ 2 arquivos |
| 8 | Hydrate in Production Line | ❌ Ausente nos poços 00002/00004/00006 |

As classes 3, 5 e 8 não ocorrem nos três poços selecionados. A expansão para outros poços é listada como trabalho futuro.

---

## Referência

PETROBRAS. **Dataset 3W** — Timely detections for more proactive and effective actions in offshore oil wells.  
Disponível em: https://github.com/petrobras/3W | Versão: 1.70.0 (abr/2026)
