# ============================================================
#  commit_and_pr.ps1
#  Commit das atualizacoes do escopo 001/002/006 + Pull Shark
#  Uso: .\commit_and_pr.ps1
#  Pre-requisito: fechar VS Code / Git Bash antes de rodar
# ============================================================

$TOKEN  = "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"
$OWNER  = "AlanPinheiro19"
$REPO   = "projeto-3w-aws"
$BRANCH = "feat/refactor-wells-001-002-006"
$BASE   = "main"

$headers = @{
    Authorization = "token $TOKEN"
    Accept        = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

function Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}
function Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "    ERRO: $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "  COMMIT + PULL SHARK - projeto 3W (escopo 001/002/006)    " -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue

# ── Passo 0: Limpar lock corrompido ──────────────────────────
Step 0 "Limpando lock files do git..."
Remove-Item -Force ".git\index.lock" -ErrorAction SilentlyContinue
Ok "Lock limpo (ou nao existia)"

# ── Passo 1: Configurar identidade ───────────────────────────
Step 1 "Configurando identidade git..."
git config user.name "AlanPinheiro19"
git config user.email "alanpinhe@gmail.com"
git remote set-url origin "https://${OWNER}:${TOKEN}@github.com/${OWNER}/${REPO}.git"
Ok "Identidade e remote configurados"

# ── Passo 2: Criar branch feature ────────────────────────────
Step 2 "Criando branch $BRANCH..."
$currentBranch = git branch --show-current
if ($currentBranch -eq $BRANCH) {
    Ok "Ja esta na branch $BRANCH"
} else {
    git checkout -b $BRANCH 2>&1
    if ($LASTEXITCODE -ne 0) {
        # Branch pode ja existir
        git checkout $BRANCH 2>&1
    }
    Ok "Branch $BRANCH ativa"
}

# ── Passo 3: Ignorar arquivos temporarios ────────────────────
Step 3 "Atualizando .gitignore..."
$ignoreExtra = @"

# Temporarios Windows
~`$*

# Pastas de sessao Claude
.claude/

# Terraform state local
terraform/.terraform/
terraform/terraform.tfstate
terraform/terraform.tfvars

# CVs pessoais
CV_*.docx
PROFILE_README.md

# Projeto separado
nlp-petroleo-rag/
"@
$currentIgnore = Get-Content ".gitignore" -Raw -ErrorAction SilentlyContinue
if ($currentIgnore -notlike "*Claude*") {
    Add-Content ".gitignore" $ignoreExtra
    Ok ".gitignore atualizado"
} else {
    Ok ".gitignore ja atualizado"
}

# ── Passo 4: Stage dos arquivos relevantes ───────────────────
Step 4 "Staging arquivos..."

# Remove temp file se rastreado
git rm --cached "~`$onograma_status_3w.docx" 2>$null

$filesToAdd = @(
    # gitignore
    ".gitignore",
    # Scripts ML corrigidos
    "scripts/train_rf_baseline.py",
    "scripts/shap_analysis.py",
    "scripts/tune_thresholds.py",
    "scripts/etl_bronze_silver_gold.py",
    "scripts/ingestao_local.py",
    "scripts/ingestao_unificada.py",
    "scripts/README.md",
    "scripts/populate_project_board.py",
    "scripts/Ingestao_well_Classes_Gith_3w/",
    # Scripts de execucao
    "run_fix_rf_shap.ps1",
    "run_gold_and_train.ps1",
    "run_pipeline_new_wells.ps1",
    "run_pipeline_new_wells.bat",
    # Resultados dos modelos
    "models/rf_baseline_metrics.json",
    "models/rf_baseline_report.txt",
    "models/rf_baseline_confusion.png",
    "models/rf_baseline_importance.png",
    "models/xgb_baseline_metrics.json",
    "models/xgb_baseline_report.txt",
    "models/lgb_baseline_metrics.json",
    "models/lgb_baseline_report.txt",
    "models/shap_results.json",
    "models/shap_global_bar.png",
    "models/shap_global_summary.png",
    "models/shap_sensor_importance.png",
    "models/shap_sensor_per_class.png",
    "models/shap_class_00_Normal.png",
    "models/shap_class_01_BSW_Abrupt.png",
    "models/shap_class_07_Scaling.png",
    "models/threshold_tuning_results.json",
    "models/threshold_tuning_report.txt",
    "models/threshold_confusion_test.png",
    "models/threshold_dhsv_curve.png",
    # Documentos e infra
    "monografia_v2.docx",
    "arquitetura_aws.png",
    "README.md",
    "config/spark_config.py",
    "docker-compose.yml",
    # Notebooks
    "notebooks/dashboard_3w.ipynb",
    "notebooks/02_metodologia_pipeline_3w.ipynb"
)

foreach ($f in $filesToAdd) {
    if (Test-Path $f) {
        git add $f 2>&1 | Out-Null
    }
}
$staged = git diff --cached --name-only | Measure-Object -Line
Ok "$($staged.Lines) arquivos staged"

# ── Passo 5: Commit ──────────────────────────────────────────
Step 5 "Criando commit..."

$commitMsg = @"
feat: refactor scope to wells 001/002/006 + fix ML pipeline

Scope change: replaced WELL-00004 (low sensor completeness) with
WELL-00001, expanding event classes from 6 to 7 (adds DHSV Closure
class 2 and Severe Slugging class 3).

ML pipeline fixes:
- train_rf_baseline.py: chunked stratified sampling (was reading
  first N rows = temporal bias -> all Normal -> F1-macro 0.0)
  RF now achieves 81.4% F1-macro baseline, 85.3% after threshold tuning
- shap_analysis.py: load only test.parquet sample (was loading
  train+val+test -> ArrowMemoryError on 199 MB file)
- tune_thresholds.py: chunked reading + fixed XGB class remapping
  guard (6-class model vs 7-class data shape mismatch)

New execution scripts:
- run_gold_and_train.ps1: full pipeline (Gold ETL + all 3 models)
- run_fix_rf_shap.ps1: re-run RF + SHAP only (skips XGB/LGB)
- run_pipeline_new_wells.ps1/.bat: entry point for new well scope

Model results (escopo 001/002/006, 7 classes, stratified splits):
  RF  baseline:  F1-macro 81.4% | tuned: 85.3% (+3.9 pp)
  XGB baseline:  F1-macro 90.0% (6-class model, jun/2026 temporal split)
  LGB baseline:  F1-macro 96.2% (6-class model, jun/2026 temporal split)

Docs updated:
- monografia_v2.docx: new section 3.11 with comparison table (Tabela 11)
  old scope 002/004/006 vs new scope 001/002/006, limitation note on
  XGB/LGB retraining as future work
- arquitetura_aws.png: generated architecture diagram (Local vs AWS)
  inserted as Figura 14 in monograph section 3.10.2
- notebooks/02_metodologia_pipeline_3w.ipynb: methodological analysis

Co-authored-by: Claude <noreply@anthropic.com>
"@

git commit -m $commitMsg
if ($LASTEXITCODE -ne 0) { Fail "Commit falhou" }
Ok "Commit criado: $(git log --oneline -1)"

# ── Passo 6: Push ────────────────────────────────────────────
Step 6 "Push para GitHub ($BRANCH)..."
git push -u origin $BRANCH --force
if ($LASTEXITCODE -ne 0) { Fail "Push falhou. Verifique token e acesso." }
Ok "Push realizado"

# ── Passo 7: Criar Pull Request ──────────────────────────────
Step 7 "Criando Pull Request..."

$prBody = @"
## Refatoracao do escopo de pocos: 001/002/006

### Motivacao
O poco WELL-00004 (escopo anterior) apresentava 100% de valores nulos em 6 dos 8 sensores,
contribuindo com dados discriminativos apenas via T-JUS-CKP e T-TPT.
A substituicao por WELL-00001 amplia o conjunto de classes de 6 para 7 eventos detectaveis.

### Correcoes no pipeline ML
| Script | Problema | Correcao |
|--------|----------|----------|
| `train_rf_baseline.py` | Primeiras N linhas = tudo Normal (temporal bias) | Amostragem estratificada em chunks de 30k |
| `shap_analysis.py` | Carregava train+val+test -> ArrowMemoryError | Carrega apenas amostra de test.parquet (20k max) |
| `tune_thresholds.py` | OOM + XGB shape mismatch (6 vs 7 classes) | Chunked reading + guard de remapeamento |

### Resultados (7 classes, splits estratificados)
| Modelo | F1-macro baseline | F1-macro tuned |
|--------|-------------------|----------------|
| Random Forest | 81.4% | **85.3%** (+3.9 pp) |
| XGBoost* | 90.0% | — |
| LightGBM* | 96.2% | — |

*Avaliados nos splits temporais do treinamento de jun/2026 (6 classes). Re-treinamento e trabalho futuro.

### Documentacao
- Secao 3.11 adicionada na monografia com tabela comparativa (Tabela 11)
- Figura 14 (arquitetura_aws.png) inserida na secao 3.10.2
- Notebook metodologico 02_metodologia_pipeline_3w.ipynb

Closes #3
"@

$prPayload = @{
    title = "feat: refactor scope to wells 001/002/006 + fix ML pipeline OOM/bias"
    body  = $prBody
    head  = $BRANCH
    base  = $BASE
} | ConvertTo-Json -Depth 3

$prResponse = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/$OWNER/$REPO/pulls" `
    -Method POST `
    -Headers $headers `
    -Body $prPayload `
    -ContentType "application/json"

$prNumber = $prResponse.number
$prUrl    = $prResponse.html_url
Ok "PR #$prNumber criado: $prUrl"

# ── Passo 8: Merge do PR (Pull Shark!) ───────────────────────
Step 8 "Mergeando PR #$prNumber (Pull Shark badge)..."
Start-Sleep -Seconds 3  # aguarda GitHub processar

$mergePayload = @{
    commit_title   = "feat: refactor scope to wells 001/002/006 (#$prNumber)"
    commit_message = "Merge pull request #$prNumber - ML pipeline fixes + docs update"
    merge_method   = "squash"
} | ConvertTo-Json

$mergeResponse = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/$OWNER/$REPO/pulls/$prNumber/merge" `
    -Method PUT `
    -Headers $headers `
    -Body $mergePayload `
    -ContentType "application/json"

Ok "PR #$prNumber mergeado com sucesso!"
Write-Host "    SHA: $($mergeResponse.sha)" -ForegroundColor DarkGray

# ── Finaliza ─────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  CONCLUIDO!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Branch : $BRANCH" -ForegroundColor White
Write-Host "  PR     : $prUrl" -ForegroundColor White
Write-Host "  Badge  : https://github.com/$OWNER (verifique Pull Shark)" -ForegroundColor White
Write-Host ""
Write-Host "  Volte ao main:" -ForegroundColor Yellow
Write-Host "  git checkout main && git pull origin main" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
