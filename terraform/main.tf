# ============================================================
# main.tf — Provider e configuração global
# Projeto TCC 3W Petrobras — eEDB-007 / USP Escola Politécnica
# ============================================================
# ⚠️  CONTA AWS ACADEMY (LabRole)
# Credenciais são temporárias (expiram ~4h).
# ANTES de qualquer terraform, exporte as variáveis do console:
#   $env:AWS_ACCESS_KEY_ID="ASIA..."
#   $env:AWS_SECRET_ACCESS_KEY="..."
#   $env:AWS_SESSION_TOKEN="..."
# ============================================================

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
  # NÃO usar backend S3 com LabRole — credenciais expiram
  # Estado local: terraform.tfstate (não commitado no git)
}

provider "aws" {
  region = var.aws_region
  # Credenciais lidas automaticamente das variáveis de ambiente
  # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN

  default_tags {
    tags = {
      Project     = "TCC-3W-Petrobras"
      Environment = var.environment
      Owner       = "Alan Pinheiro da Silva"
      ManagedBy   = "Terraform"
      Course      = "eEDB-007-USP"
    }
  }
}
