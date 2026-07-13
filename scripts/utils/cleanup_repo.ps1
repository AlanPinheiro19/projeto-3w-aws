# ============================================================
# cleanup_repo.ps1
# Remove do rastreamento git todos os artefatos que nao
# deveriam estar no repositorio.
# Os arquivos LOCAIS sao mantidos — apenas sao "desindexados".
# Execute uma unica vez, a partir da raiz do projeto.
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\.."

Write-Host "`n=== LIMPEZA DO REPOSITORIO GIT ===" -ForegroundColor Cyan
Write-Host "Diretorio: $(Get-Location)" -ForegroundColor Gray

# ------------------------------------------------------------
# 1. Artefatos gerados pelo pipeline ML (models/)
# ------------------------------------------------------------
Write-Host "`n[1/7] Removendo artefatos models/ ..." -ForegroundColor Yellow
$modelsFiles = git ls-files models/ | Where-Object { $_ -match "\.(png|json|txt|csv)$" }
foreach ($f in $modelsFiles) {
    git rm --cached $f 2>$null
    Write-Host "  - $f" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# 2. Relatórios e imagens geradas (reports/)
# ------------------------------------------------------------
Write-Host "`n[2/7] Removendo reports/ ..." -ForegroundColor Yellow
$reportsFiles = git ls-files reports/
foreach ($f in $reportsFiles) {
    git rm --cached $f 2>$null
    Write-Host "  - $f" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# 3. Figuras e arquivos de notebook (notebooks/)
# ------------------------------------------------------------
Write-Host "`n[3/7] Removendo artefatos de notebooks/ ..." -ForegroundColor Yellow
$notebookArtifacts = git ls-files notebooks/ | Where-Object {
    $_ -match "fig_.*\.png$" -or
    $_ -match "\.ipynb_checkpoints" -or
    $_ -match "notebook_teste\.py$"
}
foreach ($f in $notebookArtifacts) {
    git rm --cached -r $f 2>$null
    Write-Host "  - $f" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# 4. Documentos acadêmicos e HTMLs de rascunho
# ------------------------------------------------------------
Write-Host "`n[4/7] Removendo documentos academicos e rascunhos ..." -ForegroundColor Yellow
$docs = @(
    "artigo_cientifico_3w.docx",
    "artigo_cientifico_3w_v2.docx",
    "dashboard_editorial_blue.html",
    "mockup_dashboard_3w.html"
)
foreach ($f in $docs) {
    if (git ls-files $f) {
        git rm --cached $f 2>$null
        Write-Host "  - $f" -ForegroundColor DarkGray
    }
}

# ------------------------------------------------------------
# 5. Arquivos internos de ferramentas
# ------------------------------------------------------------
Write-Host "`n[5/7] Removendo arquivos de ferramentas (CLAUDE.md, settings.json) ..." -ForegroundColor Yellow
$toolFiles = @("CLAUDE.md", "settings.json")
foreach ($f in $toolFiles) {
    if (git ls-files $f) {
        git rm --cached $f 2>$null
        Write-Host "  - $f" -ForegroundColor DarkGray
    }
}

# ------------------------------------------------------------
# 6. Scripts de teste legados (pasta com espaco no nome)
# ------------------------------------------------------------
Write-Host "`n[6/7] Removendo scripts/testes de ingestao/ ..." -ForegroundColor Yellow
$testFiles = git ls-files "scripts/testes de ingestao/"
foreach ($f in $testFiles) {
    git rm --cached "$f" 2>$null
    Write-Host "  - $f" -ForegroundColor DarkGray
}

# ------------------------------------------------------------
# 7. Commit e push
# ------------------------------------------------------------
Write-Host "`n[7/7] Commit e push ..." -ForegroundColor Yellow

git add .gitignore
git status --short

$confirm = Read-Host "`nDeseja fazer commit e push agora? (s/n)"
if ($confirm -eq "s") {
    $msg = @'
chore: remover artefatos gerados do rastreamento git

- Remove models/*.png/json/txt (gerados pelo pipeline ML)
- Remove reports/*.png (gerados pelo dashboard)
- Remove notebooks/fig_*.png e .ipynb_checkpoints/
- Remove documentos academicos: artigo_cientifico*.docx
- Remove rascunhos HTML: dashboard_editorial, mockup_dashboard
- Remove arquivos de ferramenta: CLAUDE.md, settings.json
- Remove scripts/testes de ingestao/ (legados)
- Atualiza .gitignore com padroes abrangentes

Arquivos locais preservados - apenas desindexados do git.
'@
    git commit -m $msg
    git push origin main
    Write-Host "`n[OK] Limpeza concluida e publicada no GitHub." -ForegroundColor Green
} else {
    Write-Host "`n[OK] Arquivos desindexados localmente. Faca o commit quando quiser." -ForegroundColor Green
    Write-Host "     git commit -m 'chore: remover artefatos do rastreamento git'" -ForegroundColor Gray
    Write-Host "     git push origin main" -ForegroundColor Gray
}
