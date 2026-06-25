# Deploy AWS — Guia de Execução

Guia prático para provisionar e executar o pipeline 3W na nuvem AWS Academy.

## Pré-requisitos

- Conta AWS Academy ativa com LabRole configurado
- AWS CLI instalado e configurado (`aws configure`)
- Docker e Docker Compose instalados na instância EC2
- Chave SSH `.pem` para acesso à instância

## Infraestrutura Provisionada

| Recurso | Configuração |
|---------|-------------|
| EC2 | t3.large — 8 GB RAM, 30 GB EBS, Ubuntu 22.04 |
| S3 | Bucket `tcc-3w-datalake-raw` (raw, bronze, silver, gold, models) |
| IAM | LabRole (credenciais temporárias ~4h) |
| VPC | VPC padrão + Security Group (portas 22, 8080, 8888, 9000, 8501) |

## Passo a Passo

### 1. Iniciar Lab AWS Academy

```bash
# Exportar credenciais do Lab (AWS Academy → AWS Details → AWS CLI)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Conectar na EC2 via SSH

```bash
ssh -i alan-3w-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 3. Subir a stack Docker

```bash
cd /opt/projeto-3w
sudo git pull origin main
sudo docker compose up -d
sudo docker compose ps
```

### 4. Acessar os serviços

| Serviço | URL |
|---------|-----|
| Airflow UI | `http://<EC2_IP>:8080` — admin / admin |
| Jupyter | `http://<EC2_IP>:8888` |
| MinIO Console | `http://<EC2_IP>:9001` |
| Streamlit Dashboard | `http://<EC2_IP>:8501` |

### 5. Executar o pipeline ETL

```
Airflow UI → DAGs → dag_etl_3w_pipeline → Trigger DAG
```

Tempo estimado: ~90 min para 7,5M registros (Bronze → Silver → Gold).

### 6. Executar o treinamento ML

```
Airflow UI → DAGs → dag_ml_training → Trigger DAG
```

Tempo estimado: ~45 min (RF → XGB → LGB → Threshold Tuning → SHAP).

### 7. Sincronizar resultados para S3

```bash
aws s3 sync /opt/projeto-3w/data/gold/ s3://tcc-3w-datalake-raw/gold/
aws s3 sync /opt/projeto-3w/models/ s3://tcc-3w-datalake-raw/models/
```

## Resultados Obtidos na Nuvem

| Modelo | F1-Macro | Acurácia |
|--------|----------|----------|
| LightGBM ★ | 96,2% | 99,9% |
| XGBoost (tuned) | 95,7% | 99,3% |
| Random Forest (tuned) | 93,7% | 99,6% |

## Renovar Credenciais AWS Academy

As credenciais expiram a cada ~4h. Para renovar:

```bash
# Atualizar variáveis de ambiente com novas credenciais do Lab
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
```

## Observações

- O arquivo `.env` com credenciais nunca deve ser commitado no Git
- Modelos `.joblib` estão no `.gitignore` — use S3 para versionamento
- O swap de 4GB no EC2 é necessário para o treinamento ML (`/swapfile`)
