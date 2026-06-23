# ============================================================
# s3.tf — Buckets S3 (camadas Medallion + modelos)
# Custo: ~$0.02/GB/mês — 3 poços ≈ 500MB = < $0.01/mês
# ============================================================

locals {
  buckets = {
    raw    = "${var.s3_bucket_prefix}-raw"
    bronze = "${var.s3_bucket_prefix}-bronze"
    silver = "${var.s3_bucket_prefix}-silver"
    gold   = "${var.s3_bucket_prefix}-gold"
    models = "${var.s3_bucket_prefix}-models"
    logs   = "${var.s3_bucket_prefix}-logs"
  }
}

resource "aws_s3_bucket" "data" {
  for_each      = local.buckets
  bucket        = each.value
  force_destroy = true  # Permite terraform destroy sem esvaziar manualmente
  tags          = { Layer = each.key }
}

# Bloquear acesso público em todos os buckets
resource "aws_s3_bucket_public_access_block" "data" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.data[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Criptografia AES256 em todos os buckets
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.data[each.key].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle: remover logs antigos após 30 dias (economia de storage)
resource "aws_s3_bucket_lifecycle_configuration" "logs_cleanup" {
  bucket = aws_s3_bucket.data["logs"].id
  rule {
    id     = "cleanup-logs"
    status = "Enabled"
    expiration { days = 30 }
    filter { prefix = "" }
  }
}
