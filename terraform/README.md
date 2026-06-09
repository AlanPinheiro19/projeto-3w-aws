DataSet esta sendo criado localmente para testes antes de incluir no S3
Estou trabalhando apenas com o Poço -00002 para efeitos de aprendizageme redução de volumetria de daddos
Informações adicionais :
 - Bucket S3 tcc-ptbr-3w-datalake-2026
            -/raw
            -/bronze
            -/silver
            -/gold

Observação de Projeto: arquivos PARQUET com timestamp no final que foram testado, ocasionam erro na leitura pelo ATHENA. Gerar um único arquiivivo por classe para evitar ERROS Futuros
 