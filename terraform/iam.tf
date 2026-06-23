# ============================================================
# iam.tf — AWS Academy LabRole
# ⚠️  NÃO é possível criar recursos IAM com LabRole.
# O LabInstanceProfile já existe e é atribuído automaticamente
# à EC2 — dá acesso ao S3 sem chaves hardcoded.
# ============================================================

# Referência ao perfil pré-existente do AWS Academy
# (não criamos, apenas referenciamos no ec2.tf)
# iam_instance_profile = "LabInstanceProfile"
#
# Permissões já concedidas pelo LabRole:
#  - S3: leitura/escrita em qualquer bucket da conta
#  - EC2: operações de instância
#  - CloudWatch: logs e métricas
#
# Se precisar validar o LabRole via CLI:
#   aws iam get-role --role-name LabRole
