"""
Script: etl_bronze_silver_gold.py
Descricao: Pipeline ETL local com tres camadas de maturidade de dados.

  RAW  -> BRONZE : tipagem, schema padrao, metadados de origem
  BRONZE -> SILVER: limpeza de nulos, remocao de duplicatas, validacao de ranges
  SILVER -> GOLD  : rolling features, lag, normalizacao Z-score, divisao treino/val/teste

Uso (dentro do container Docker ou com pandas instalado):
  python scripts/etl_bronze_silver_gold.py [--step all|bronze|silver|gold] [--verbose]

Dependencias:
  pip install pandas pyarrow numpy scikit-learn
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.spark_config import (
    RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
    setup_directories, logger,
)

# ---------------------------------------------------------------------------
# Constantes do dataset 3W
# ---------------------------------------------------------------------------

SENSOR_COLUMNS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP",
    "T-JUS-CKP", "P-JUS-CKGL", "T-JUS-CKGL", "QGL",
]

TARGET_COLUMN = "class"
TIMESTAMP_COLUMN = "timestamp"

EVENT_LABELS = {
    0: "Normal",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Undefined",
}

# Ranges fisicos validos dos sensores (baseado na documentacao do 3W)
# Valores fora desses limites sao marcados como nulos na camada Silver
SENSOR_PHYSICAL_RANGES = {
    "P-PDG":        (0.0,   1000.0),
    "P-TPT":        (0.0,   1000.0),
    "T-TPT":        (-10.0, 200.0),
    "P-MON-CKP":    (0.0,   1000.0),
    "T-JUS-CKP":    (-10.0, 200.0),
    "P-JUS-CKGL":   (0.0,   1000.0),
    "T-JUS-CKGL":   (-10.0, 200.0),
    "QGL":          (0.0,   5000.0),
}

ROLLING_WINDOWS = [5, 10, 30]
LAG_STEPS = [1, 3, 5]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ---------------------------------------------------------------------------
# Utilitarios comuns
# ---------------------------------------------------------------------------

def load_parquets_from_dir(directory: Path, exclude_names=None) -> pd.DataFrame:
    """Carrega todos os .parquet de um diretorio em um unico DataFrame."""
    exclude_names = exclude_names or []
    files = [
        f for f in directory.glob("*.parquet")
        if f.name not in exclude_names
    ]
    if not files:
        logger.warning("Nenhum arquivo Parquet encontrado em: %s", directory)
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as exc:
            logger.error("Falha ao ler %s: %s", f.name, exc)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Carregados %d arquivos | %d linhas x %d colunas de %s",
                len(files), *df.shape, directory)
    return df


def save_parquet(df: pd.DataFrame, path: Path, name: str) -> None:
    """Salva DataFrame como Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Salvo: %s | %d linhas | %.2f MB", name, len(df), size_mb)


def print_class_distribution(df: pd.DataFrame, label: str) -> None:
    """Exibe distribuicao de classes no log."""
    if TARGET_COLUMN not in df.columns and "classe_evento" not in df.columns:
        return
    col = TARGET_COLUMN if TARGET_COLUMN in df.columns else "classe_evento"
    dist = df[col].value_counts().sort_index()
    logger.info("Distribuicao de classes em [%s]:", label)
    for cls, cnt in dist.items():
        nome = EVENT_LABELS.get(int(cls), "?")
        pct = cnt / len(df) * 100
        logger.info("  Classe %d (%s): %d linhas (%.1f%%)", cls, nome, cnt, pct)


# ---------------------------------------------------------------------------
# CAMADA BRONZE
# Objetivo: schema padrao, tipagem correta, metadados de origem
# Fonte: data/processed/ (arquivos ja com ID_Poco e classe_evento)
# ---------------------------------------------------------------------------

def run_bronze() -> pd.DataFrame:
    """
    Leitura da camada Processed e criacao da Bronze.

    Processamento chunked (arquivo por arquivo) com escrita incremental via
    pyarrow.ParquetWriter. Nao carrega todos os dados na RAM de uma vez,
    permitindo processar datasets de qualquer tamanho em instancias com
    memoria limitada (ex: t3.large 8 GB com 500+ arquivos / 9 M linhas).

    Transformacoes por arquivo:
    - Padronizacao de timestamp para datetime
    - Conversao de sensores para float64
    - Normalizacao de classe para int8
    - Remocao de registros com class < 0
    - Adicao de label textual e metadado bronze_at

    Retorna DataFrame vazio: a camada Silver le diretamente do arquivo
    gerado em disco (BRONZE_DIR/bronze_3w_all.parquet).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("=" * 60)
    logger.info("BRONZE - iniciando (modo chunked por arquivo)")
    logger.info("=" * 60)

    from config.spark_config import PROCESSED_DIR

    exclude = {"all_classes_combined.parquet", "all_classes_combined.csv"}
    parquet_files = sorted([
        f for f in PROCESSED_DIR.glob("*.parquet")
        if f.name not in exclude
    ])

    if not parquet_files:
        logger.error("Sem dados em processed/ para criar Bronze.")
        return pd.DataFrame()

    logger.info("Encontrados %d arquivos Parquet em %s", len(parquet_files), PROCESSED_DIR)

    keep_cols = [TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN, "nome_evento"] + SENSOR_COLUMNS
    bronze_at_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out = BRONZE_DIR / "bronze_3w_all.parquet"

    writer = None
    total_rows = 0
    removed_total = 0

    for i, fpath in enumerate(parquet_files):
        try:
            df = pd.read_parquet(fpath)
        except Exception as exc:
            logger.error("Falha ao ler %s: %s", fpath.name, exc)
            continue

        # Normaliza timestamp
        for ts_col in ["timestamp", "index", "Timestamp"]:
            if ts_col in df.columns:
                df = df.rename(columns={ts_col: TIMESTAMP_COLUMN})
                break
        if TIMESTAMP_COLUMN in df.columns:
            df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

        # Sensores -> float64
        for col in SENSOR_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            else:
                df[col] = np.nan

        # Classe -> int8
        if "classe_evento" in df.columns:
            df[TARGET_COLUMN] = df["classe_evento"].fillna(-1).astype("int8")
        elif TARGET_COLUMN in df.columns:
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(-1).astype("int8")
        else:
            df[TARGET_COLUMN] = np.int8(-1)

        # Remove invalidos
        before = len(df)
        df = df[df[TARGET_COLUMN] >= 0].copy()
        removed_total += before - len(df)

        if df.empty:
            continue

        # Label textual
        if "nome_evento" not in df.columns:
            df["nome_evento"] = df[TARGET_COLUMN].map(EVENT_LABELS).fillna("Desconhecido")

        # Seleciona colunas + metadado
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()
        df["bronze_at"] = bronze_at_str

        # Escrita incremental: abre writer na primeira tabela valida
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out), table.schema, compression="snappy")
        writer.write_table(table)

        total_rows += len(df)

        # Log de progresso a cada 50 arquivos
        if (i + 1) % 50 == 0 or (i + 1) == len(parquet_files):
            logger.info(
                "Bronze: %d/%d arquivos processados | %d linhas acumuladas",
                i + 1, len(parquet_files), total_rows,
            )

        # Libera memoria explicitamente
        del df, table

    if writer:
        writer.close()
        logger.info(
            "Bronze concluido: %d linhas gravadas em %s (removidos %d invalidos)",
            total_rows, out, removed_total,
        )
    else:
        logger.error("Nenhum dado valido encontrado em processed/.")

    # Retorna DataFrame vazio: run_silver() lerá do arquivo em disco
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CAMADA SILVER
# Objetivo: dados limpos e confiaveis para analise
# Fonte: data/bronze/
# ---------------------------------------------------------------------------

def run_silver(df_bronze: pd.DataFrame = None) -> pd.DataFrame:
    """
    Limpeza e validacao dos dados da camada Bronze.

    Transformacoes:
    - Forward fill + backward fill por poco (por classe de evento)
    - Invalida leituras fora dos ranges fisicos dos sensores
    - Remove duplicatas de timestamp por poco
    - Remove registros onde todos os sensores sao nulos
    """
    logger.info("=" * 60)
    logger.info("SILVER - iniciando")
    logger.info("=" * 60)

    if df_bronze is None or df_bronze.empty:
        df_bronze = load_parquets_from_dir(BRONZE_DIR)

    if df_bronze.empty:
        logger.error("Sem dados na camada Bronze.")
        return pd.DataFrame()

    df = df_bronze.copy()

    # 1. Invalida leituras fora dos ranges fisicos
    for sensor, (vmin, vmax) in SENSOR_PHYSICAL_RANGES.items():
        if sensor not in df.columns:
            continue
        mask_invalid = (df[sensor] < vmin) | (df[sensor] > vmax)
        invalid_count = mask_invalid.sum()
        if invalid_count > 0:
            logger.info("Silver: %d leituras invalidas em %s -> substituidas por NaN",
                        invalid_count, sensor)
            df.loc[mask_invalid, sensor] = np.nan

    # 2. Forward fill + backward fill por poco e classe
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]
    sort_cols = [TIMESTAMP_COLUMN] if TIMESTAMP_COLUMN in df.columns else []

    if partition_cols and sort_cols:
        df = df.sort_values(partition_cols + sort_cols)
        for sensor in SENSOR_COLUMNS:
            if sensor not in df.columns:
                continue
            df[sensor] = (
                df.groupby(partition_cols)[sensor]
                .transform(lambda s: s.ffill().bfill())
            )

    # 3. Remove duplicatas de timestamp por poco
    if TIMESTAMP_COLUMN in df.columns and "ID_Poco" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["ID_Poco", TIMESTAMP_COLUMN], keep="first")
        dup_removed = before - len(df)
        if dup_removed > 0:
            logger.info("Silver: %d duplicatas de timestamp removidas.", dup_removed)

    # 4. Remove linhas onde TODOS os sensores ainda sao nulos
    sensor_cols_present = [c for c in SENSOR_COLUMNS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=sensor_cols_present, how="all")
    all_null_removed = before - len(df)
    if all_null_removed > 0:
        logger.info("Silver: %d linhas com todos os sensores nulos removidas.",
                    all_null_removed)

    # 5. Preenche nulos residuais com mediana global por classe (fallback)
    for sensor in sensor_cols_present:
        if df[sensor].isna().sum() > 0:
            if TARGET_COLUMN in df.columns:
                df[sensor] = df.groupby(TARGET_COLUMN)[sensor].transform(
                    lambda s: s.fillna(s.median())
                )
            df[sensor] = df[sensor].fillna(df[sensor].median())

    # Metadados
    df["silver_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Silver: %d linhas x %d colunas", *df.shape)
    print_class_distribution(df, "Silver")

    out = SILVER_DIR / "silver_3w_all.parquet"
    save_parquet(df, out, "silver_3w_all.parquet")
    return df


# ---------------------------------------------------------------------------
# CAMADA GOLD
# Objetivo: dataset ML-ready com features engineered e normalizado
# Fonte: data/silver/
# ---------------------------------------------------------------------------

def _safe_col(name: str) -> str:
    """Converte nome de sensor para nome de coluna valido (sem hifens)."""
    return name.replace("-", "_")


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, std, min, max por janela e por poco.

    Estrategia de memoria: processa UM SENSOR POR VEZ e concatena ao df
    imediatamente, liberando o dict antes de passar para o proximo sensor.
    Pico de alocacao = ~12 colunas (1 sensor x 4 stats x 3 janelas) em vez
    de 96 colunas de uma vez. Features sao salvas em float32 (metade da RAM).
    """
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]

    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        safe = _safe_col(sensor)
        grp = df.groupby(partition_cols)[sensor] if partition_cols else df[sensor]
        new_cols: dict = {}
        for w in ROLLING_WINDOWS:
            new_cols[f"{safe}_roll_mean_{w}"] = (
                grp.transform(lambda s, _w=w: s.rolling(_w, min_periods=1).mean())
                .astype("float32")
            )
            new_cols[f"{safe}_roll_std_{w}"] = (
                grp.transform(lambda s, _w=w: s.rolling(_w, min_periods=1).std().fillna(0))
                .astype("float32")
            )
            new_cols[f"{safe}_roll_min_{w}"] = (
                grp.transform(lambda s, _w=w: s.rolling(_w, min_periods=1).min())
                .astype("float32")
            )
            new_cols[f"{safe}_roll_max_{w}"] = (
                grp.transform(lambda s, _w=w: s.rolling(_w, min_periods=1).max())
                .astype("float32")
            )
        # Atribuição direta evita pd.concat e a consolidação de blocos (OOM)
        for col_name, series in new_cols.items():
            df[col_name] = series
        del new_cols

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variaveis de lag temporal por poco.
    Processa um sensor por vez para limitar o pico de memoria (~3 colunas/sensor).
    Features em float32.
    """
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]

    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        safe = _safe_col(sensor)
        for lag in LAG_STEPS:
            if partition_cols:
                col = df.groupby(partition_cols)[sensor].shift(lag).astype("float32")
            else:
                col = df[sensor].shift(lag).astype("float32")
            df[f"{safe}_lag_{lag}"] = col
            del col

    return df


def create_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de variacao (delta) entre instantes consecutivos por poco.
    Processa um sensor por vez. Feature em float32.
    """
    partition_cols = [c for c in ["ID_Poco", TARGET_COLUMN] if c in df.columns]

    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        safe = _safe_col(sensor)
        if partition_cols:
            delta = df.groupby(partition_cols)[sensor].diff().astype("float32")
        else:
            delta = df[sensor].diff().astype("float32")
        df[f"{safe}_delta"] = delta
        del delta

    return df


def normalize_features(df: pd.DataFrame, feature_cols: list, train_mask: pd.Series) -> tuple:
    """
    Normalizacao Z-score calculada APENAS no conjunto de treino.
    Opera in-place (renomeia colunas para _norm) para evitar duplicar memoria.
    Retorna (df_normalizado, dict_stats) para uso posterior na inferencia.
    """
    stats = {}
    rename_map = {}

    for col in feature_cols:
        if col not in df.columns:
            continue
        train_vals = df.loc[train_mask, col]
        mean_val = float(train_vals.mean())
        std_val  = float(train_vals.std())
        if pd.isna(std_val) or std_val == 0:
            std_val = 1.0
        stats[col] = {"mean": mean_val, "std": std_val}
        # Normaliza a coluna no lugar — sem duplicar o DataFrame
        df[col] = (df[col] - mean_val) / std_val
        rename_map[col] = f"{col}_norm"

    df = df.rename(columns=rename_map)
    return df, stats


def run_gold(df_silver: pd.DataFrame = None) -> pd.DataFrame:
    """
    Engenharia de features e preparacao do dataset ML-ready.

    Estrategia de memoria (evita OOM com datasets > 1M linhas):
    - Passo 1: features computadas por poco -> parquets temporarios (sem acumulacao)
    - Passo 2: stats de normalizacao calculadas do conjunto de treino via acumulacao
               de sum/sumsq (memoria constante, independente do tamanho)
    - Passo 3: normalizacao aplicada por poco e saida escrita incrementalmente

    Transformacoes:
    - Rolling statistics (mean, std, min, max) janelas 5, 10, 30
    - Variaveis de lag (1, 3, 5 passos)
    - Taxa de variacao (delta) por sensor
    - Normalizacao Z-score (stats calculadas so no treino, globalmente)
    - Divisao temporal: treino 70% / validacao 15% / teste 15%
    - Salva train.parquet, val.parquet, test.parquet em data/gold/
    """
    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("=" * 60)
    logger.info("GOLD - iniciando (modo eficiente em memoria)")
    logger.info("=" * 60)

    if df_silver is None or df_silver.empty:
        df_silver = load_parquets_from_dir(SILVER_DIR)

    if df_silver.empty:
        logger.error("Sem dados na camada Silver.")
        return pd.DataFrame()

    # Ordena por poco + timestamp
    sort_cols = [c for c in ["ID_Poco", TIMESTAMP_COLUMN] if c in df_silver.columns]
    if sort_cols:
        df_silver = df_silver.sort_values(sort_cols).reset_index(drop=True)

    META_COLS = {TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN,
                 "nome_evento", "bronze_at", "silver_at"}

    # ------------------------------------------------------------------
    # PASSO 1: Feature engineering por poco -> parquets temporarios
    # Cada poco e processado e liberado imediatamente (pico: 1 poco na RAM)
    # ------------------------------------------------------------------
    temp_dir = GOLD_DIR / "_temp_features"
    temp_dir.mkdir(parents=True, exist_ok=True)

    wells = sorted(df_silver["ID_Poco"].unique()) if "ID_Poco" in df_silver.columns else [None]
    logger.info("Gold - Passo 1: features em %d pocos -> parquets temporarios...", len(wells))

    total_rows = 0
    feature_cols: list = []

    for i, well in enumerate(wells, 1):
        if well is not None:
            df_w = df_silver[df_silver["ID_Poco"] == well].copy()
        else:
            df_w = df_silver.copy()

        df_w = create_rolling_features(df_w)
        df_w = create_lag_features(df_w)
        df_w = create_delta_features(df_w)

        # Remove NaN de janelas iniciais
        fcols = [c for c in df_w.columns if c not in META_COLS]
        df_w = df_w.dropna(subset=fcols, how="any")

        if df_w.empty:
            del df_w
            continue

        if not feature_cols:
            feature_cols = fcols  # captura lista de features do primeiro poco

        temp_path = temp_dir / f"{i:05d}.parquet"
        df_w.to_parquet(temp_path, index=False)
        total_rows += len(df_w)
        del df_w
        gc.collect()

        if i % 10 == 0 or i == len(wells):
            logger.info("  %d/%d pocos escritos (%d linhas acumuladas)", i, len(wells), total_rows)

    # Libera df_silver completamente
    del df_silver
    gc.collect()

    if total_rows == 0 or not feature_cols:
        logger.error("Nenhuma linha valida apos feature engineering.")
        return pd.DataFrame()

    logger.info("Gold - Passo 1 concluido: %d linhas totais, %d features.", total_rows, len(feature_cols))

    # ------------------------------------------------------------------
    # PASSO 2: Stats de normalizacao (treino) via acumulacao sum/sumsq
    # Memoria constante: apenas 3 dicts de floats, independente do volume
    # ------------------------------------------------------------------
    train_end = int(total_rows * TRAIN_RATIO)
    val_end   = int(total_rows * (TRAIN_RATIO + VAL_RATIO))

    logger.info("Gold - Passo 2: computando stats de normalizacao do treino (%d linhas)...", train_end)

    f_sum   = {col: 0.0 for col in feature_cols}
    f_sum2  = {col: 0.0 for col in feature_cols}
    f_count = {col: 0   for col in feature_cols}
    seen = 0

    for pf in sorted(temp_dir.glob("*.parquet")):
        if seen >= train_end:
            break
        df_w = pd.read_parquet(pf, columns=feature_cols)
        take = min(len(df_w), train_end - seen)
        df_t = df_w.iloc[:take]
        for col in feature_cols:
            if col in df_t.columns:
                vals = df_t[col].to_numpy(dtype="float64", na_value=np.nan)
                vals = vals[~np.isnan(vals)]
                f_sum[col]   += float(vals.sum())
                f_sum2[col]  += float((vals ** 2).sum())
                f_count[col] += len(vals)
        seen += take
        del df_w, df_t
        gc.collect()

    scaler_stats: dict = {}
    for col in feature_cols:
        n = f_count[col] or 1
        mean_val = f_sum[col] / n
        var_val  = max(f_sum2[col] / n - mean_val ** 2, 0.0)
        std_val  = var_val ** 0.5 if var_val > 0 else 1.0
        scaler_stats[col] = {"mean": round(mean_val, 8), "std": round(std_val, 8)}

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = GOLD_DIR / "scaler_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(scaler_stats, fh, indent=2)
    logger.info("Scaler stats salvas: %s", stats_path)

    # ------------------------------------------------------------------
    # PASSO 3: Normalizacao por poco + escrita incremental (pyarrow)
    # Pico de RAM = 1 poco normalizado; saida escrita sem concat global
    # ------------------------------------------------------------------
    logger.info("Gold - Passo 3: normalizando e escrevendo train/val/test...")

    keep_meta = [c for c in [TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN, "nome_evento"]
                 if c in pd.read_parquet(sorted(temp_dir.glob("*.parquet"))[0], nrows=0 if False else None).columns]

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    train_writer = val_writer = test_writer = None
    schema = None
    written_rows = {"train": 0, "val": 0, "test": 0}
    cumulative = 0

    for pf in sorted(temp_dir.glob("*.parquet")):
        df_w = pd.read_parquet(pf)

        # Aplica normalizacao in-place
        rename_map = {}
        for col in feature_cols:
            if col in df_w.columns:
                s = scaler_stats[col]
                df_w[col] = (df_w[col] - s["mean"]) / s["std"]
                rename_map[col] = f"{col}_norm"
        df_w = df_w.rename(columns=rename_map)

        norm_cols = [c for c in df_w.columns if c.endswith("_norm")]
        final_cols = [c for c in keep_meta if c in df_w.columns] + norm_cols
        df_w = df_w[final_cols].copy()
        df_w["gold_at"] = now_str

        # Determina split por posicao global
        well_len = len(df_w)
        well_start = cumulative
        well_end   = cumulative + well_len

        # Parcelas de treino/val/test dentro deste poco
        t_start = max(0,         train_end - well_start)
        v_start = max(0,         val_end   - well_start)

        df_train_w = df_w.iloc[: min(t_start, well_len)]
        df_val_w   = df_w.iloc[t_start : min(v_start, well_len)]
        df_test_w  = df_w.iloc[v_start :]

        # Escreve incrementalmente com pyarrow
        for split_df, split_name in [
            (df_train_w, "train"), (df_val_w, "val"), (df_test_w, "test")
        ]:
            if split_df.empty:
                continue
            table = pa.Table.from_pandas(split_df, preserve_index=False)
            if split_name == "train":
                if train_writer is None:
                    train_writer = pq.ParquetWriter(GOLD_DIR / "train.parquet", table.schema)
                train_writer.write_table(table)
            elif split_name == "val":
                if val_writer is None:
                    val_writer = pq.ParquetWriter(GOLD_DIR / "val.parquet", table.schema)
                val_writer.write_table(table)
            else:
                if test_writer is None:
                    test_writer = pq.ParquetWriter(GOLD_DIR / "test.parquet", table.schema)
                test_writer.write_table(table)
            written_rows[split_name] += len(split_df)

        cumulative += well_len
        del df_w, df_train_w, df_val_w, df_test_w
        gc.collect()

    # Fecha writers
    for w in [train_writer, val_writer, test_writer]:
        if w is not None:
            w.close()

    # Remove arquivos temporarios
    for pf in temp_dir.glob("*.parquet"):
        pf.unlink()
    temp_dir.rmdir()

    logger.info("Gold: divisao temporal:")
    logger.info("  Treino:    %d linhas (%.0f%%)", written_rows["train"], TRAIN_RATIO * 100)
    logger.info("  Validacao: %d linhas (%.0f%%)", written_rows["val"],   VAL_RATIO   * 100)
    logger.info("  Teste:     %d linhas (%.0f%%)", written_rows["test"],  TEST_RATIO  * 100)
    logger.info("  Features:  %d colunas normalizadas", len(feature_cols))

    # Retorna amostra do treino para compatibilidade com chamadas downstream
    sample = pd.read_parquet(GOLD_DIR / "train.parquet").head(1000)
    return sample


# ---------------------------------------------------------------------------
# GOLD - Funcoes por-poco (dynamic task mapping no Airflow)
# ---------------------------------------------------------------------------

def list_gold_wells() -> list:
    """Lista IDs de pocos disponiveis na camada Silver.
    Leitura eficiente: carrega apenas a coluna ID_Poco via pyarrow.
    Retorna lista ordenada de strings para uso no XCom do Airflow.
    """
    import pyarrow.parquet as pq

    files = sorted(SILVER_DIR.glob("*.parquet"))
    if not files:
        logger.warning("list_gold_wells: nenhum parquet em %s", SILVER_DIR)
        return []

    wells: set = set()
    for f in files:
        try:
            tbl = pq.read_table(str(f), columns=["ID_Poco"])
            wells.update(
                str(v) for v in tbl["ID_Poco"].to_pylist() if v is not None
            )
        except Exception as exc:
            logger.error("Erro ao ler %s: %s", f.name, exc)

    result = sorted(wells)
    logger.info("list_gold_wells: %d pocos encontrados.", len(result))
    return result


def load_well_from_silver(well_id: str) -> pd.DataFrame:
    """Carrega APENAS as linhas de um poco da Silver usando filtro pyarrow.
    O predicate pushdown do pyarrow le so as linhas necessarias do disco,
    evitando carregar os 7M+ de linhas do arquivo completo na RAM.
    """
    import pyarrow.parquet as pq

    files = sorted(SILVER_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            tbl = pq.read_table(str(f), filters=[("ID_Poco", "=", well_id)])
            if tbl.num_rows > 0:
                dfs.append(tbl.to_pandas())
        except Exception as exc:
            logger.error("load_well_from_silver: erro em %s: %s", f.name, exc)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Converte colunas de sensor para float32 (metade da RAM vs float64)
    for col in SENSOR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    logger.info("Poco %s: %d linhas carregadas da Silver (sensores em float32).", well_id, len(df))
    return df


def run_gold_process_well(well_id: str) -> int:
    """Feature engineering para um unico poco.
    Carrega apenas as linhas do poco (pyarrow filter), aplica rolling/lag/delta
    e salva em data/gold/_temp_features/well_<id>.parquet.
    Libera a RAM ao final via gc.collect().
    Retorna o numero de linhas escritas (0 se o poco estiver vazio).
    """
    import gc

    logger.info("=" * 50)
    logger.info("Gold-Poco [%s] - iniciando", well_id)

    META_COLS = {TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN,
                 "nome_evento", "bronze_at", "silver_at"}

    # Carrega apenas este poco (eficiente em memoria via pyarrow filter)
    df = load_well_from_silver(well_id)
    if df.empty:
        logger.warning("Gold-Poco [%s]: sem dados. Ignorando.", well_id)
        return 0

    # Ordena por timestamp antes das janelas de rolling
    if TIMESTAMP_COLUMN in df.columns:
        df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    # Feature engineering (funcoes ja otimizadas com pd.concat)
    df = create_rolling_features(df)
    gc.collect()
    df = create_lag_features(df)
    gc.collect()
    df = create_delta_features(df)
    gc.collect()

    # Preenche NaN coluna por coluna para evitar alocar matriz booleana 3M×128
    # e copia de todas as features de uma vez (pico de ~2 GB evitado).
    fcols = [c for c in df.columns if c not in META_COLS]
    nan_total = 0
    for c in fcols:
        n_nan = int(df[c].isna().sum())
        if n_nan > 0:
            df[c] = df[c].fillna(np.float32(0.0))
            nan_total += n_nan
    if nan_total > 0:
        logger.info("Gold-Poco [%s]: %d NaN preenchidos com 0 nas features.", well_id, nan_total)
    gc.collect()

    # Descarta apenas linhas onde TODOS os sensores originais sao NaN (dados ausentes)
    sensor_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
    before = len(df)
    if sensor_cols:
        df = df.dropna(subset=sensor_cols, how="all")
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Gold-Poco [%s]: %d linhas removidas (todos sensores NaN).", well_id, dropped)

    if df.empty:
        logger.warning("Gold-Poco [%s]: vazio apos dropna.", well_id)
        return 0

    # Salva parquet temporario
    temp_dir = GOLD_DIR / "_temp_features"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_id = (str(well_id)
               .replace("/", "_").replace("\\", "_")
               .replace(" ", "_").replace(":", "_"))
    temp_path = temp_dir / f"well_{safe_id}.parquet"
    df.to_parquet(temp_path, index=False)

    n = len(df)
    logger.info("Gold-Poco [%s]: %d linhas -> %s", well_id, n, temp_path.name)
    del df
    gc.collect()
    return n


def run_gold_finalize() -> pd.DataFrame:
    """Finaliza a camada Gold: calcula stats de normalizacao e escreve os splits.

    Le os parquets temporarios gerados por run_gold_process_well(), calcula as
    estatisticas Z-score globais sobre o conjunto de treino via acumulacao
    sum/sumsq (memoria constante), normaliza cada poco e escreve
    train.parquet / val.parquet / test.parquet incrementalmente via pyarrow.

    Remove os arquivos temporarios ao final.
    """
    import gc
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("=" * 60)
    logger.info("Gold-Final - iniciando")
    logger.info("=" * 60)

    META_COLS = {TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN,
                 "nome_evento", "bronze_at", "silver_at", "gold_at"}

    temp_dir = GOLD_DIR / "_temp_features"
    temp_files = sorted(temp_dir.glob("well_*.parquet"))

    if not temp_files:
        logger.error("Gold-Final: nenhum parquet temporario em %s", temp_dir)
        return pd.DataFrame()

    logger.info("Gold-Final: %d arquivos temporarios encontrados.", len(temp_files))

    # Detecta feature_cols e keep_meta a partir do primeiro arquivo
    schema_pa = pq.read_schema(str(temp_files[0]))
    all_cols = schema_pa.names
    feature_cols = [c for c in all_cols if c not in META_COLS]
    keep_meta = [c for c in [TIMESTAMP_COLUMN, "ID_Poco", TARGET_COLUMN, "nome_evento"]
                 if c in all_cols]

    # Conta total de linhas para determinar split (sem carregar dados)
    total_rows = sum(pq.read_metadata(str(f)).num_rows for f in temp_files)
    train_end = int(total_rows * TRAIN_RATIO)
    val_end   = int(total_rows * (TRAIN_RATIO + VAL_RATIO))

    logger.info("Gold-Final: %d linhas | treino=%d | val=%d | teste=%d",
                total_rows, train_end, val_end - train_end, total_rows - val_end)
    logger.info("Gold-Final: %d features a normalizar.", len(feature_cols))

    # ------------------------------------------------------------------
    # PASSO A: Stats Z-score via sum/sumsq (memoria constante)
    # ------------------------------------------------------------------
    logger.info("Gold-Final: Passo A - computando stats de normalizacao...")

    f_sum   = {col: 0.0 for col in feature_cols}
    f_sum2  = {col: 0.0 for col in feature_cols}
    f_count = {col: 0   for col in feature_cols}
    seen = 0

    for pf in temp_files:
        if seen >= train_end:
            break
        df_w = pd.read_parquet(pf, columns=feature_cols)
        take = min(len(df_w), train_end - seen)
        df_t = df_w.iloc[:take]
        for col in feature_cols:
            if col in df_t.columns:
                vals = df_t[col].to_numpy(dtype="float64", na_value=np.nan)
                vals = vals[~np.isnan(vals)]
                f_sum[col]   += float(vals.sum())
                f_sum2[col]  += float((vals ** 2).sum())
                f_count[col] += len(vals)
        seen += take
        del df_w, df_t
        gc.collect()

    scaler_stats: dict = {}
    for col in feature_cols:
        n = f_count[col] or 1
        mean_v = f_sum[col] / n
        var_v  = max(f_sum2[col] / n - mean_v ** 2, 0.0)
        std_v  = var_v ** 0.5 if var_v > 0 else 1.0
        scaler_stats[col] = {"mean": round(mean_v, 8), "std": round(std_v, 8)}

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = GOLD_DIR / "scaler_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(scaler_stats, fh, indent=2)
    logger.info("Scaler stats salvas: %s", stats_path)

    # ------------------------------------------------------------------
    # PASSO B: Normalizacao + escrita incremental (pyarrow)
    # ------------------------------------------------------------------
    logger.info("Gold-Final: Passo B - normalizando e escrevendo parquets...")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    train_writer = val_writer = test_writer = None
    written_rows = {"train": 0, "val": 0, "test": 0}
    cumulative = 0

    for pf in temp_files:
        df_w = pd.read_parquet(pf)

        # Normaliza cada feature in-place e renomeia para _norm
        rename_map = {}
        for col in feature_cols:
            if col in df_w.columns:
                s = scaler_stats[col]
                df_w[col] = (df_w[col] - s["mean"]) / s["std"]
                rename_map[col] = f"{col}_norm"
        df_w = df_w.rename(columns=rename_map)

        norm_cols  = [c for c in df_w.columns if c.endswith("_norm")]
        final_cols = [c for c in keep_meta if c in df_w.columns] + norm_cols
        df_w = df_w[final_cols].copy()
        df_w["gold_at"] = now_str

        # Determina parcelas de treino/val/test dentro deste poco
        well_len = len(df_w)
        t_cut = max(0, train_end - cumulative)
        v_cut = max(0, val_end   - cumulative)

        splits = [
            (df_w.iloc[:min(t_cut, well_len)],      "train"),
            (df_w.iloc[t_cut:min(v_cut, well_len)], "val"),
            (df_w.iloc[v_cut:],                     "test"),
        ]

        for split_df, split_name in splits:
            if split_df.empty:
                continue
            tbl = pa.Table.from_pandas(split_df, preserve_index=False)
            if split_name == "train":
                if train_writer is None:
                    train_writer = pq.ParquetWriter(GOLD_DIR / "train.parquet", tbl.schema)
                train_writer.write_table(tbl)
            elif split_name == "val":
                if val_writer is None:
                    val_writer = pq.ParquetWriter(GOLD_DIR / "val.parquet", tbl.schema)
                val_writer.write_table(tbl)
            else:
                if test_writer is None:
                    test_writer = pq.ParquetWriter(GOLD_DIR / "test.parquet", tbl.schema)
                test_writer.write_table(tbl)
            written_rows[split_name] += len(split_df)

        cumulative += well_len
        del df_w
        gc.collect()

    # Fecha writers
    for w in [train_writer, val_writer, test_writer]:
        if w is not None:
            w.close()

    # Remove arquivos temporarios
    for pf in temp_files:
        try:
            pf.unlink()
        except Exception:
            pass
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    logger.info("Gold-Final concluido:")
    logger.info("  Treino:    %d linhas (%.0f%%)", written_rows["train"], TRAIN_RATIO * 100)
    logger.info("  Validacao: %d linhas (%.0f%%)", written_rows["val"],   VAL_RATIO   * 100)
    logger.info("  Teste:     %d linhas (%.0f%%)", written_rows["test"],  TEST_RATIO  * 100)
    logger.info("  Features:  %d colunas normalizadas", len(feature_cols))

    # Retorna amostra do treino para compatibilidade
    train_path = GOLD_DIR / "train.parquet"
    if train_path.exists():
        return pd.read_parquet(train_path).head(1000)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(step: str = "all", verbose: bool = False) -> None:
    """
    Executa o pipeline ETL completo ou uma etapa especifica.

    Args:
        step: "all" | "bronze" | "silver" | "gold"
        verbose: se True, exibe mais detalhes no log
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_directories()

    start = datetime.now(timezone.utc)
    logger.info("Pipeline ETL iniciado | etapa: %s | %s", step, start.strftime("%Y-%m-%d %H:%M:%S"))

    df_bronze = pd.DataFrame()
    df_silver = pd.DataFrame()

    if step in ("all", "bronze"):
        df_bronze = run_bronze()

    if step in ("all", "silver"):
        df_silver = run_silver(df_bronze if not df_bronze.empty else None)

    if step in ("all", "gold"):
        run_gold(df_silver if not df_silver.empty else None)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("Pipeline ETL concluido em %.1f segundos.", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ETL Bronze/Silver/Gold - Dataset 3W",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python etl_bronze_silver_gold.py --step all
  python etl_bronze_silver_gold.py --step bronze
  python etl_bronze_silver_gold.py --step gold_list
  python etl_bronze_silver_gold.py --step gold_poco --well WELL_00001
  python etl_bronze_silver_gold.py --step gold_final
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "bronze", "silver", "gold",
                 "gold_list", "gold_poco", "gold_final"],
        default="all",
        help=(
            "Etapa a executar. "
            "gold_list: lista pocos da Silver (JSON). "
            "gold_poco: feature engineering de um poco (requer --well). "
            "gold_final: normaliza e escreve train/val/test."
        ),
    )
    parser.add_argument(
        "--well",
        type=str,
        default=None,
        help="ID do poco para --step gold_poco (ex: WELL_00001)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Nivel de log DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_directories()

    if args.step == "gold_list":
        # Imprime lista JSON de pocos no stdout (lida pelo DAG via subprocess)
        wells = list_gold_wells()
        print(json.dumps(wells))

    elif args.step == "gold_poco":
        if not args.well:
            parser.error("--well e obrigatorio quando --step=gold_poco")
        n = run_gold_process_well(args.well)
        sys.exit(0 if n >= 0 else 1)

    elif args.step == "gold_final":
        run_gold_finalize()

    else:
        run_pipeline(step=args.step, verbose=args.verbose)
