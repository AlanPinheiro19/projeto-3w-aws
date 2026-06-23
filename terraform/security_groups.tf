# ============================================================
# security_groups.tf — Apenas SG da EC2 (sem RDS)
# ============================================================

resource "aws_security_group" "ec2" {
  name        = "${var.project_name}-ec2-sg"
  description = "Acesso restrito ao IP do desenvolvedor"
  vpc_id      = aws_vpc.main.id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "SSH - apenas seu IP"
  }

  # Airflow Web UI
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "Airflow UI"
  }

  # Jupyter Lab
  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "Jupyter Lab"
  }

  # Streamlit Dashboard
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "Streamlit"
  }

  # MinIO Console (acesso ao browser da interface)
  ingress {
    from_port   = 9001
    to_port     = 9001
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
    description = "MinIO Console"
  }

  # Saída irrestrita (GitHub API, pip, Docker Hub)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-ec2-sg" }
}
