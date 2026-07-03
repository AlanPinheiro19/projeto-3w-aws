@echo off
REM ============================================================
REM  run_pipeline_new_wells.bat
REM  Pipeline completo 3W com novo escopo de pocos
REM  Pocos: WELL-00001, WELL-00002, WELL-00006
REM  Ambiente: conda "3W"
REM
REM  Uso: clique duplo ou execute de qualquer pasta
REM ============================================================

REM Navega para a raiz do projeto (dois niveis acima de scripts\pipeline\)
cd /d "%~dp0..\.."

setlocal
set ENV=3W
set CONDA=conda run -n "%ENV%" --no-capture-output

echo ============================================================
echo  PIPELINE 3W - WELL-00001 / WELL-00002 / WELL-00006
echo ============================================================
echo.

REM Passo 0: Limpa dados derivados (mantém data\raw intacto)
echo [0/5] Limpando dados anteriores...
if exist data\gold     rmdir /s /q data\gold
if exist data\silver   rmdir /s /q data\silver
if exist data\bronze   rmdir /s /q data\bronze
if exist data\processed rmdir /s /q data\processed
echo     OK - diretorios limpos.
echo.

REM Passo 1: Ingestao dos pocos do novo escopo
echo [1/5] Ingestao dos pocos 00001, 00002, 00006 (todas as classes)...
%CONDA% python scripts\ingestao_unificada.py --wells 00001 00002 00006
if %ERRORLEVEL% neq 0 (
    echo ERRO na ingestao. Abortando.
    exit /b 1
)
echo     OK - ingestao concluida.
echo.

REM Passo 2: ETL em tres camadas (Bronze -> Silver -> Gold)
echo [2/5] ETL: Bronze / Silver / Gold...
%CONDA% python scripts\etl_bronze_silver_gold.py
if %ERRORLEVEL% neq 0 (
    echo ERRO no ETL. Abortando.
    exit /b 1
)
echo     OK - ETL concluido. Dados em data\gold\
echo.

REM Passo 3: Treinamento dos modelos baseline
echo [3/5] Treinando modelos...

echo   [3a] Random Forest...
%CONDA% python scripts\train_rf_baseline.py
if %ERRORLEVEL% neq 0 echo   AVISO: RF com erro - continuando...

echo   [3b] XGBoost...
%CONDA% python scripts\train_xgb_baseline.py
if %ERRORLEVEL% neq 0 echo   AVISO: XGB com erro - continuando...

echo   [3c] LightGBM...
%CONDA% python scripts\train_lgb_baseline.py
if %ERRORLEVEL% neq 0 echo   AVISO: LGB com erro - continuando...

echo     OK - modelos treinados. Resultados em models\
echo.

REM Passo 4: Ajuste de thresholds por classe
echo [4/5] Threshold Tuning por classe...
%CONDA% python scripts\tune_thresholds.py
if %ERRORLEVEL% neq 0 echo   AVISO: Threshold tuning com erro - continuando...
echo     OK
echo.

REM Passo 5: Analise SHAP de interpretabilidade
echo [5/5] SHAP Analysis (interpretabilidade)...
%CONDA% python scripts\shap_analysis.py
if %ERRORLEVEL% neq 0 echo   AVISO: SHAP com erro - continuando...
echo     OK
echo.

echo ============================================================
echo  PIPELINE CONCLUIDO
echo  Verifique models\ para os novos resultados
echo  Compare com baseline anterior (00002/00004/00006)
echo ============================================================
pause
