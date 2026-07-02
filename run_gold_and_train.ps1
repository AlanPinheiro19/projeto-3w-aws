# Pipeline 3W - Retomada a partir do Gold
# Bronze e Silver ja estao prontos.
# Este script executa: Gold (por poco) -> Treino -> Threshold -> SHAP
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
Write-Host "  RETOMADA DO PIPELINE 3W (Gold + Treino)" -ForegroundColor Blue
Write-Host "  Bronze: OK (206 MB)  |  Silver: OK (90 MB)" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue

# Passo 0: Limpa apenas o Gold
Step 0 6 "Limpando Gold anterior..."
if (Test-Path "data\gold") {
    Remove-Item -Recurse -Force "data\gold"
    Write-Host "    Removido: data\gold" -ForegroundColor DarkGray
}
Ok "Gold limpo"

# Passo 1: Descobre os IDs reais dos pocos no Silver (ex: 00001, 00002, 00006)
Step 1 6 "Lendo IDs de pocos do Silver..."
$wellsJson = conda run -n $ENV_NAME python scripts\etl_bronze_silver_gold.py --step gold_list 2>$null
Write-Host "    IDs encontrados: $wellsJson" -ForegroundColor DarkGray

try {
    $wells = $wellsJson | ConvertFrom-Json
} catch {
    Write-Host "    Fallback: usando IDs padrao 00001, 00002, 00006" -ForegroundColor Yellow
    $wells = @("00001", "00002", "00006")
}

if ($wells.Count -eq 0) {
    Write-Host "    Fallback: usando IDs padrao 00001, 00002, 00006" -ForegroundColor Yellow
    $wells = @("00001", "00002", "00006")
}
Ok "Pocos: $($wells -join ', ')"

# Passo 2: Gold por poco (memoria eficiente)
Step 2 6 "Gold - Feature engineering por poco..."
foreach ($well in $wells) {
    Write-Host "    Processando poco $well ..." -ForegroundColor DarkGray
    conda run -n $ENV_NAME --no-capture-output python scripts\etl_bronze_silver_gold.py --step gold_poco --well $well
    if ($LASTEXITCODE -ne 0) { Fail "gold_poco $well falhou." }
    Ok "Poco $well concluido"
}

# Passo 3: Gold final (normalizacao Z-score + split 70/15/15)
Step 3 6 "Gold - Finalizacao (normalizacao + split train/val/test)..."
conda run -n $ENV_NAME --no-capture-output python scripts\etl_bronze_silver_gold.py --step gold_final
if ($LASTEXITCODE -ne 0) { Fail "gold_final falhou." }

if (-not (Test-Path "data\gold\train.parquet")) {
    Fail "train.parquet nao foi gerado. Verifique os logs."
}
Ok "Gold finalizado - train/val/test.parquet gerados"

# Passo 4: Treino dos modelos
Step 4 6 "Treinando modelos (RF / XGB / LGB)..."

Write-Host "    [4a] Random Forest..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output python scripts\train_rf_baseline.py
if ($LASTEXITCODE -ne 0) { Warn "RF com erro - continuando..." }

Write-Host "    [4b] XGBoost..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output python scripts\train_xgb_baseline.py
if ($LASTEXITCODE -ne 0) { Warn "XGB com erro - continuando..." }

Write-Host "    [4c] LightGBM..." -ForegroundColor DarkGray
conda run -n $ENV_NAME --no-capture-output python scripts\train_lgb_baseline.py
if ($LASTEXITCODE -ne 0) { Warn "LGB com erro - continuando..." }

Ok "Modelos treinados - resultados em models\"

# Passo 5: Threshold Tuning
Step 5 6 "Threshold tuning por classe..."
conda run -n $ENV_NAME --no-capture-output python scripts\tune_thresholds.py
if ($LASTEXITCODE -ne 0) { Warn "Threshold tuning com erro - continuando..." }
Ok "Threshold tuning concluido"

# Passo 6: SHAP
Step 6 6 "Instalando SHAP e executando analise..."
conda run -n $ENV_NAME --no-capture-output pip install shap -q
conda run -n $ENV_NAME --no-capture-output python scripts\shap_analysis.py
if ($LASTEXITCODE -ne 0) { Warn "SHAP com erro - verifique logs" }
Ok "SHAP concluida"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  PIPELINE CONCLUIDO" -ForegroundColor Green
Write-Host "  Baseline anterior: LGB F1-macro 96,19% (pocos 002/004/006)" -ForegroundColor Green
Write-Host "  Verifique models\ para comparar os novos resultados" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
