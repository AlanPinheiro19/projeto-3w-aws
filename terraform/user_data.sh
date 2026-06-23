#!/bin/bash
# ============================================================
# user_data.sh — Bootstrap EC2: Docker + stack 3W (sem RDS)
# PostgreSQL roda em Docker para economizar (sem RDS)
# ============================================================
set -euo pipefail
exec > /var/log/user_data.log 2>&1
echo "=== Bootstrap 3W TCC iniciado: $(date) ==="

# ── 1. Atualizar SO ───────────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y -q
apt-get install -y git curl wget unzip jq python3 python3-pip python3-venv awscli

# ── 2. Instalar Docker ────────────────────────────────────────────────────────
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable docker && systemctl start docker
usermod -aG docker ubuntu

# ── 3. Clonar repositório ────────────────────────────────────────────────────
mkdir -p /opt/projeto-3w && cd /opt/projeto-3w
git clone https://github.com/AlanPinheiro19/projeto-3w-aws.git . 2>/dev/null || \
  { git config --global --add safe.directory /opt/projeto-3w && git pull 2>/dev/null || true; }

# ── 4. Criar .env ──────────────────────────────────────────────────────────────
# Credenciais S3 via LabInstanceProfile — SEM chaves hardcoded
cat > /opt/projeto-3w/.env << ENVEOF
# Gerado pelo Terraform bootstrap — NÃO edite manualmente
AWS_DEFAULT_REGION=${aws_region}
AWS_REGION=${aws_region}

# S3 Buckets (acesso via LabInstanceProfile — sem AWS_ACCESS_KEY_ID)
S3_BUCKET_RAW=${s3_bucket_prefix}-raw
S3_BUCKET_BRONZE=${s3_bucket_prefix}-bronze
S3_BUCKET_SILVER=${s3_bucket_prefix}-silver
S3_BUCKET_GOLD=${s3_bucket_prefix}-gold
S3_BUCKET_MODELS=${s3_bucket_prefix}-models
S3_BUCKET_LOGS=${s3_bucket_prefix}-logs
DATA_BACKEND=s3

# PostgreSQL em Docker (sem RDS — economia de ~$15/mês)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=${postgres_password}
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:${postgres_password}@postgres/airflow

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__WEBSERVER__SECRET_KEY=$(openssl rand -hex 32)
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=${airflow_admin_password}

# Projeto
PROJECT_NAME=${project_name}
ENVEOF
chmod 600 /opt/projeto-3w/.env

# ── 5. Dependências Python ────────────────────────────────────────────────────
cd /opt/projeto-3w
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q 2>/dev/null || true

# ── 6. Subir stack Docker ─────────────────────────────────────────────────────
cd /opt/projeto-3w
docker compose --env-file .env up -d 2>/dev/null || \
  docker-compose --env-file .env up -d 2>/dev/null || true

# ── 7. Aguardar Airflow e criar usuário admin ─────────────────────────────────
sleep 90
docker compose exec -T airflow-webserver \
  airflow users create \
  --username admin \
  --password "${airflow_admin_password}" \
  --firstname Alan --lastname Pinheiro \
  --role Admin --email alanpinhe@gmail.com 2>/dev/null || true

MY_IP=$(curl -s ifconfig.me 2>/dev/null || echo "IP_DESCONHECIDO")
echo "=== Bootstrap concluído: $(date) ==="
echo "=== Airflow: http://$MY_IP:8080 ==="
