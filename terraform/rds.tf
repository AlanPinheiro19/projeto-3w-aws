# ============================================================
# rds.tf — DESATIVADO para economizar orçamento ($50 total)
# ============================================================
# RDS db.t3.micro custaria ~$15/mês — impacto significativo
# no orçamento de $50 para 2 meses de projeto.
#
# Solução adotada: PostgreSQL rodando em Docker na própria EC2.
# Configurado via user_data.sh → .env → docker-compose.yml:
#   POSTGRES_HOST=postgres
#   POSTGRES_PORT=5432
#   AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:PASS@postgres/airflow
#
# Para reativar o RDS (pós-TCC, conta com budget maior):
#   1. Adicione variáveis rds_* no variables.tf
#   2. Adicione subnets privadas e SG do RDS no vpc.tf/security_groups.tf
#   3. Descomente o resource abaixo
#   4. Atualize connection string no user_data.sh
# ============================================================

# resource "aws_db_instance" "airflow" {
#   identifier     = "${var.project_name}-airflow-db"
#   engine         = "postgres"
#   engine_version = "15.5"
#   instance_class = "db.t3.micro"
#   ... (desativado — use PostgreSQL no Docker)
# }
