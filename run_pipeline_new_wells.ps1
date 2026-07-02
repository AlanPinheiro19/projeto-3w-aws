# Pipeline 3W - Reexecucao com novos pocos
# Pocos: WELL-00001, WELL-00002, WELL-00006
# Ambiente: conda 3W

$ENV_NAME = "3W"

function Step($n, $total, $msg) {
    Write-Host ""
    Write-Host "[$n/$total] $msg" -ForegroundColor Cyan
}
function Ok($msg)   { Write-Host "    OK - $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    AVISO: $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "    ERRO: $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "  PIPELINE 3W - WELL-00001 / WELL-00002 / WELL-00006" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue

# Passo 0: Limpa dados derivados (mantem data\raw intacto)
Step 0 5 "Limpando dados derivados anteriores..."
foreach ($dir in @("data\gold", "data\silver", "data\bronze", "data\processed")) {
    if (Test-Path $dir) {
        Remove-Item -Recurse -Force $dir
        Write-Host "    Removido: $dir" -ForegroundColor DarkGray
    }
}
Ok "Diretorios limpos"

# Passo 1: Ingestao
Step 1 5 "Ingestao dos pocos 00001, 00002, 00006..."
conda run -n $ENV_NAME --no-capture-output python scripts\ingestao_unificada.py --wells 00001 00002 00006
if ($LASTEXITCODE -ne 0) { Fail "Ingestao falhou. Verifique conexao com GitHub ou GITHUB_TOKEN." }
Ok "Ingestao concluida"

# Passo 2: ETL Bronze / Silver / Gold
Step 2 5 "ETL: Bronze -> Silver -> Gold..."
conda run -n $ENV_NAME --no-capture-output python scripts\etl_bronze_silver_gold.py
if ($LASTEXITCODE -ne 0) { Fail "ETL falhou." }
Ok "ETL concluido - dados em data\gold\"

# Passo 3: Treinamento
Step 3 5 "Treinando modelos (RF / XGB / LGB)..."

Write-Host "    [3a] Random Forest..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output python scripts\train_rf_baseline.py
if ($LASTEXITCODE -ne 0) { Warn "RF com erro - continuando..." }

Write-Host "    [3b] XGBoost..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output python scripts\train_xgb_baseline.py
if ($LASTEXITCODE -ne 0) { Warn "XGB com erro - continuando..." }

Write-Host "    [3c] LightGBM..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output python scripts\train_lgb_baseline.py
if ($LASTEXITCODE -ne 0) { Warn "LGB com erro - continuando..." }

Ok "Modelos treinados - resultados em models\"

# Passo 4: Threshold Tuning
Step 4 5 "Threshold tuning por classe..."
conda run -n $ENV_NAME --no-capture-output python scripts\tune_thresholds.py
if ($LASTEXITCODE -ne 0) { Warn "Threshold tuning com erro - continuando..." }
Ok "Threshold tuning concluido"

# Passo 5: SHAP
Step 5 5 "SHAP Analysis..."
conda run -n $ENV_NAME --no-capture-output python scripts\shap_analysis.py
if ($LASTEXITCODE -ne 0) { Warn "SHAP com erro - continuando..." }
Ok "SHAP concluida"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  PIPELINE CONCLUIDO" -ForegroundColor Green
Write-Host "  Baseline anterior: LGB F1-macro 96,19% (pocos 002/004/006)" -ForegroundColor Green
Write-Host "  Verifique models\ para os novos resultados" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
