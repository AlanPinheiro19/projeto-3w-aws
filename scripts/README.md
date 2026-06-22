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



O que o script faz em cada camada:

Bronze :
 lê data/processed/, padroniza schema (timestamp, float64 nos sensores, int8 na classe), descarta registros sem classe válida
Silver :
 invalida leituras fora dos ranges físicos dos sensores, forward/backward fill por poço, remove duplicatas de timestamp, preenche nulos residuais com mediana por classe
Gold :
 rolling stats (mean/std/min/max, janelas 5/10/30), lag features (1,3,5), delta, normalização Z-score calculada só no treino, divide temporalmente 70/15/15 e salva train.parquet, val.parquet, test.parquet + scaler_stats.json



Resultado Obitido :

1  Distribuição de classes — barras + pizza com imbalance ratio
2  Boxplots dos 8 sensores por classe
3  Heatmap de correlação (seaborn)
4  Média e desvio padrão dos sensores por classe (grouped bars)
5  Poder discriminativo dos sensores (ranking)
6  Percentual de nulos com alerta em 5%
7  Distribuição de classes nos splits treino/val/teste
8  Histogramas sobrepostos das features normalizadas (Z-score)
9  Série temporal do poço com mais classes distintas
10 Resumo executivo com métricas-chave

git branch -M main
git add scripts/
git commit -m "Primeiro commit: scripts de ingestão do TCC"
git remote add origin https://github.com/AlanPinheiro19/pece-usp-tcc.git
git push -u origin main
 

 Atividades 18/06/2026
# Ações imediatas
1. Resolver crash Docker Desktop/WSL2  →  wsl --shutdown  →  reiniciar Docker Desktop [ x ]
2. Copiar ETL fix + dag_gold_rebuild.py para D:\...\projeto-3w-aws\  via PowerShell [ x ]
3. Trigger dag_gold_rebuild no Airflow UI (http://localhost:8080) [ x ]
4. Validar Gold: train/val/test.parquet com linhas > 0 e 141 colunas [ x ]
5. Executar python scripts/train_rf_baseline.py  (após Gold validado) [ x ]


Cronograma v4 entregue. Aqui a leitura dos resultados do RF:

O que os números dizem:
O gap grande entre validação (F1mac=0.51) e teste (F1mac=0.24) é esperado e tem causa conhecida: o split 70/15/15 sequencial faz o modelo "ver" timestamps de eventos raros durante o treino que são similares aos do teste, mas na validação o conjunto ainda fica próximo ao treino. O modelo aprendeu bem a classe Normal (que domina), mas falha nas anomalias raras (classes 1–8 com proporção <1%).