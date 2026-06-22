# projeto-3w-aws

**Sistema para Detecção e Classificação de Eventos Indesejáveis em Poços de Petróleo Offshore**

Trabalho de Conclusão de Curso (TCC) — Especialização em Engenharia de Dados e Big Data  
USP Escola Politécnica — Programa eEDB-007  
Autor: Alan Pinheiro da Silva | 2026

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
│   ├── dag_etl_3w_pipeline.py   # ET