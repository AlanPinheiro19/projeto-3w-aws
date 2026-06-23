# ============================================================
# outputs.tf — Valores exibidos após terraform apply
# ============================================================

output "ec2_public_ip" {
  value       = aws_eip.main.public_ip
  description = "IP fixo da EC2"
}

output "ssh_command" {
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_eip.main.public_ip}"
  description = "Comando SSH para acessar a EC2"
}

output "airflow_url" {
  value = "http://${aws_eip.main.public_ip}:8080"
}

output "jupyter_url" {
  value = "http://${aws_eip.main.public_ip}:8888"
}

output "streamlit_url" {
  value = "http://${aws_eip.main.public_ip}:8501"
}

output "s3_buckets" {
  value = { for k, v in aws_s3_bucket.data : k => v.bucket }
}

output "custo_estimado" {
  value = "EC2 t3.large: $0.0832/hr. 6h/dia x 60 dias = ~$30. S3 3 pocos (~500MB) = ~$0.01/mes. Total: ~$30-35 dentro do orcamento de $50."
}
