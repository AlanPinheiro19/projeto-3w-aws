# pece-usp-tcc
Detecção de Anomalias em Sensores de Poços Submarinos com Machine Learning

# Visão Geral do Projeto
Este repositório documenta o Trabalho de Conclusão de Curso (TCC) apresentado ao curso de Especialização em Engenharia de Dados e Big Data, cujo objetivo foi detectar anomalias operacionais em sensores de poços submarinos utilizando técnicas de Machine Learning em ambiente de nuvem AWS.

A abordagem adotada simula um pipeline completo de dados em larga escala, desde a ingestão até a predição de eventos indevidos, com foco na redução de riscos operacionais, ambientais e dos custos associados a intervenções corretivas em poços offshore.

# Abordagem Técnica e Arquitetura
O projeto foi estruturado seguindo boas práticas de engenharia de dados e metodologias de projetos de grandes volumes de dados, com os seguintes componentes principais:

# 1. Ingestão de Dados
Coleta simulada ou real de dados provenientes de sensores instalados em poços submarinos (pressão, temperatura, vazão, vibração, etc.)

Utilização de serviços AWS para captura e armazenamento escalável

# 2. Processamento e Classificação
Limpeza, transformação e enriquecimento dos dados brutos

Classificação inicial de eventos operacionais normais e anômalos

# 3. Modelagem com Machine Learning
Desenvolvimento e treinamento de modelos de detecção de anomalias

Algoritmos explorados: Isolation Forest, Random Forest, XGBoost 

# 4. Análise e Proposição
Avaliação da acurácia dos modelos em identificar eventos operacionais indevidos

Proposta de um framework preditivo para alerta precoce de anomalias

# Benefícios Esperados da Solução
Redução de riscos operacionais – detecção antecipada de falhas ou comportamentos anômalos

Minimização de impactos ambientais – prevenção de vazamentos ou operações fora do controle

Diminuição de custos e tempo de correção – atuação preditiva em vez de reativa

Aplicabilidade em cenários reais de produção offshore


# Tecnologias e Serviços Utilizados
Camada	Ferramentas / Serviços AWS
Ingestão	AWS Kinesis, S3, Lambda
Armazenamento	S3 (Data Lake), DynamoDB ou RDS
Processamento	AWS Glue, EMR, Spark
Machine Learning	SageMaker (ou scikit-learn / TensorFlow local)
Orquestração	Step Functions, Airflow
Visualização	QuickSight / Grafana


# Principais Aprendizados Consolidados

Ao longo do curso e deste TCC, foi possível aprender e aplicar na prática:

Conceitos fundamentais de computação em nuvem aplicados a dados massivos

Pipelines completos de Big Data na AWS

Estratégias de classificação e detecção de anomalias com Machine Learning

Como integrar domínio de engenharia de petróleo com ciência de dados
