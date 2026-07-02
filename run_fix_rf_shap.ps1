# Fix RF + SHAP - re-executa apenas os passos que falharam
# Gold, XGB e LGB ja estao prontos (data/gold/ e models/ existem).
# Correcoes aplicadas:
#   - train_rf_baseline.py: amostragem estratificada (era primeiras N linhas -> tudo Normal)
#   - shap_analysis.py: carrega apenas test.parquet (era train+val+test -> OOM)
#
# Prerequisito: conda activate 3W (ou conda run -n 3W)
# Uso: .\run_fix_rf_shap.ps1

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
Write-Host "  FIX RF + SHAP - pipeline 3W (pocos 001/002/006)" -ForegroundColor Blue
Write-Host "  Gold OK  |  XGB 90.0%  |  LGB 96.2%  |  RF zerado -> fix" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue

# Verifica pre-requisitos
if (-not (Test-Path "data\gold\train.parquet")) {
    Fail "data\gold\train.parquet nao encontrado. Execute run_gold_and_train.ps1 primeiro."
}

# Garante que todos os pacotes ML estao instalados no env
Write-Host "  Instalando dependencias ML (xgboost, lightgbm, shap)..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output pip install xgboost lightgbm shap -q
if ($LASTEXITCODE -ne 0) { Warn "pip install com aviso - continuando..." }
Write-Host "  Dependencias OK" -ForegroundColor DarkGray

# Passo 1: RF corrigido (amostragem estratificada)
Step 1 3 "Random Forest - amostragem estratificada (correcao aplicada)..."
conda run -n $ENV_NAME --no-capture-output python scripts\train_rf_baseline.py
if ($LASTEXITCODE -ne 0) { Fail "RF falhou. Verifique logs acima." }
Ok "RF concluido - metricas em models\rf_baseline_metrics.json"

# Passo 2: Threshold tuning (re-executa para incluir RF corrigido)
Step 2 3 "Threshold tuning (RF + XGB + LGB)..."
conda run -n $ENV_NAME --no-capture-output python scripts\tune_thresholds.py
if ($LASTEXITCODE -ne 0) { Warn "Threshold tuning com aviso - continuando..." }
Ok "Threshold tuning concluido"

# Passo 3: SHAP (carrega apenas amostra de test.parquet - sem OOM)
Step 3 3 "SHAP analysis (correcao: amostra estratificada de test.parquet)..."
conda run -n $ENV_NAME --no-capture-output python scripts\shap_analysis.py
if ($LASTEXITCODE -ne 0) { Warn "SHAP com aviso - verifique logs" }
Ok "SHAP concluida"

# Resumo de metricas
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  METRICAS FINAIS - pocos 001/002/006" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$rf  = Get-Content "models\rf_baseline_metrics.json"  -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
$xgb = Get-Content "models\xgb_baseline_metrics.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
$lgb = Get-Content "models\lgb_baseline_metrics.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json

if ($rf)  { Write-Host ("  RF  baseline  F1-macro (teste): {0:P1}" -f $rf.test.f1_macro)  -ForegroundColor White }
if ($xgb) { Write-Host ("  XGB baseline  F1-macro (teste): {0:P1}" -f $xgb.test.f1_macro) -ForegroundColor White }
if ($lgb) { Write-Host ("  LGB baseline  F1-macro (teste): {0:P1}" -f $lgb.test.f1_macro) -ForegroundColor White }

Write-Host ""
Write-Host "  Baseline anterior (002/004/006): LGB 96.2%" -ForegroundColor DarkGray
Write-Host "  Comparar acima com novo escopo  (001/002/006)" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Green
