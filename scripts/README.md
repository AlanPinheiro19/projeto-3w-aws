DataSet esta sendo criado localmente para testes antes de incluir no S3
Estou trabalhando apenas com o Poço -00002 para efeitos de aprendizageme redução de volumetria de daddos
Informações adicionais :
 - Bucket S3 tcc-ptbr-3w-datalake-2026
            -/raw
            -/bronze
            -/silver
            -/gold

Observação de Projeto: 
    -arquivos PARQUET com timestamp no final que foram testado, ocasionam erro na leitura pelo ATHENA. Gerar um único arquiivivo por classe para evitar ERROS Futuros

    -A política atual do usuário não permite algumas operações como escrever dados, essa política é atribuída automaticamente ao iniciar o sistema, afim de impedir escalonamentos e altos custos na utlização do ambiente 

    - Como alternativa de contorno os scripts de ingestão serão executados localmente e depois os arquivo carregados de forma manual no Bucket S3.  
    - Foi tulizado dados reais inciados por WELL e desconsiderados dados desenhados a mão ou simulados e todas as Classes ( 0 a 9)  algumas classes não possuiem dados para os poços selecionados
    
    - Ingestão feita no Bucket s3 camada Raw s3://tcc-ptbr-3w-datalake-2026/raw/


CHECKLIST DE EXECUÇÃO
Passo	Ação	Status
1	Verificar dados no S3	[ x ]
2	Criar Database no Glue	[ x ]
3	Criar IAM Role	[ LabRole ]
4	Executar Crawler (raw)	[ x ]
5	Verificar tabelas no Data Catalog	[ x ]
6	Criar Glue ETL Job	[ x ]
7	Executar ETL Job	[ x ]
8	Verificar dados no Athena	[ x ]
9	Executar Crawler (silver)	[ x ]




git branch -M main
git add scripts/
git commit -m "Primeiro commit: scripts de ingestão do TCC"
git remote add origin https://github.com/AlanPinheiro19/pece-usp-tcc.git
git push -u origin main
 