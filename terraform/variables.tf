# ============================================================
# variables.tf — Variáveis do projeto (LabRole + Budget $50)
# ============================================================

variable "aws_region" {
  description = "Região AWS"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  type    = string
  default = "tcc"
}

variable "project_name" {
  type    = string
  default = "3w-petrobras"
}

# ── Rede ──────────────────────────────────────────────────────────────────────
variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR da subnet pública (EC2)"
  type        = string
  default     = "10.0.1.0/24"
}

variable "availability_zone" {
  description = "AZ única (1 subnet pública — sem RDS, não precisa de 2 AZs)"
  type        = string
  default     = "us-east-1a"
}

# ── EC2 ───────────────────────────────────────────────────────────────────────
variable "ec2_instance_type" {
  description = <<-EOT
    Tipo da instância.
    t3.large  = 8GB RAM, $0.0832/hr → desligar quando não usar
    t3.medium = 4GB RAM, $0.0464/hr → mais econômico mas pode ser limitado para Airflow
    Recomendado: t3.large (desligar = economiza ~$0.50/hora parada)
  EOT
  type    = string
  default = "t3.large"
}

variable "ec2_ami_id" {
  description = "AMI Ubuntu 22.04 LTS us-east-1 — verifique a mais recente antes do deploy"
  type        = string
  default     = "ami-0c7217cdde317cfec"
}

variable "ec2_volume_size_gb" {
  description = "Disco EBS (mínimo 30GB para Docker + 3 poços ~500MB)"
  type        = number
  default     = 30
}

variable "key_pair_name" {
  description = "Nome do Key Pair criado no Console AWS"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "Seu IP público + /32. Nunca use 0.0.0.0/0"
  type        = string
}

# ── S3 ────────────────────────────────────────────────────────────────────────
variable "s3_bucket_prefix" {
  description = "Prefixo globalmente único (ex: 3w-tcc-alan-2026)"
  type        = string
  default     = "3w-tcc-alan"
}

# ── Airflow ───────────────────────────────────────────────────────────────────
variable "airflow_admin_password" {
  description = "Senha do admin do Airflow"
  type        = string
  sensitive   = true
}

# ── PostgreSQL (Docker — sem RDS para economizar) ─────────────────────────────
variable "postgres_password" {
  description = "Senha do PostgreSQL rodando em Docker (não RDS)"
  type        = string
  sensitive   = true
}
