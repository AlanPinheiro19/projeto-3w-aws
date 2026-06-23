# ============================================================
# ec2.tf — EC2 com LabInstanceProfile (AWS Academy)
# Sem RDS: PostgreSQL roda em Docker na própria EC2
# ============================================================

resource "aws_instance" "main" {
  ami                    = var.ec2_ami_id
  instance_type          = var.ec2_instance_type
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ec2.id]

  # LabInstanceProfile pré-existente no AWS Academy
  # Dá acesso ao S3 sem chaves hardcoded
  iam_instance_profile   = "LabInstanceProfile"

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.ec2_volume_size_gb
    delete_on_termination = true
    encrypted             = true
    tags = { Name = "${var.project_name}-ec2-disk" }
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    project_name           = var.project_name
    aws_region             = var.aws_region
    s3_bucket_prefix       = var.s3_bucket_prefix
    airflow_admin_password = var.airflow_admin_password
    postgres_password      = var.postgres_password
  })

  tags = { Name = "${var.project_name}-ec2" }

  lifecycle {
    ignore_changes = [user_data]
  }
}

# Elastic IP — fixo, não muda quando a EC2 é iniciada/parada
# Custo: GRATUITO enquanto associado a uma EC2 rodando
# Custo: $0.005/hora quando a EC2 está PARADA — liberar se parar > 2 dias
resource "aws_eip" "main" {
  instance = aws_instance.main.id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-eip" }
}
