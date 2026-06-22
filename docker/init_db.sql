-- init_db.sql
-- Inicializacao do banco de dados PostgreSQL para o projeto 3W Offshore Events
-- Executado automaticamente na primeira inicializacao do container postgres

-- ---------------------------------------------------------------------------
-- Schema principal
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS well_events;

-- ---------------------------------------------------------------------------
-- Tabela de pocos cadastrados
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS well_events.wells (
    id              SERIAL PRIMARY KEY,
    well_id         VARCHAR(10)  NOT NULL UNIQUE,  -- ex: "00002"
    well_name       VARCHAR(100),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Tabela de tipos de evento (referencia)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS well_events.event_types (
    classe          INTEGER      PRIMARY KEY,
    nome_curto      VARCHAR(50)  NOT NULL,
    descricao       TEXT
);

INSERT INTO well_events.event_types (classe, nome_curto, descricao) VALUES
    (0, 'Normal',                   'Operacao normal sem anomalias'),
    (1, 'BSW_Abrupt',               'Abrupt Increase of BSW'),
    (2, 'DHSV_Closure',             'Spurious Closure of DHSV'),
    (3, 'Severe_Slugging',          'Severe Slugging'),
    (4, 'Flow_Instability',         'Flow Instability'),
    (5, 'Productivity_Loss',        'Rapid Productivity Loss'),
    (6, 'PCK_Restriction',          'Quick Restriction in PCK'),
    (7, 'PCK_Scaling',              'Scaling in PCK'),
    (8, 'Hydrate_Production',       'Hydrate in Production Line'),
    (9, 'Undefined',                'Undefined/Other')
ON CONFLICT (classe) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Tabela de execucoes de ingestao (rastreabilidade)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS well_events.ingestion_runs (
    id              SERIAL PRIMARY KEY,
    run_date        DATE         NOT NULL DEFAULT CURRENT_DATE,
    started_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMP,
    classes         TEXT[],
    wells           TEXT[],
    files_found     INTEGER      DEFAULT 0,
    files_downloaded INTEGER     DEFAULT 0,
    files_failed    INTEGER      DEFAULT 0,
    total_rows      BIGINT       DEFAULT 0,
    status          VARCHAR(20)  DEFAULT 'running',  -- running | completed | failed
    notes           TEXT
);

-- ---------------------------------------------------------------------------
-- Tabela de metricas de modelos ML
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS well_events.model_metrics (
    id              SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    model_name      VARCHAR(50)  NOT NULL,  -- random_forest | xgboost | lightgbm
    trained_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    n_train         INTEGER,
    n_test          INTEGER,
    n_features      INTEGER,
    f1_weighted     NUMERIC(6,4),
    f1_macro        NUMERIC(6,4),
    accuracy        NUMERIC(6,4),
    precision_weighted NUMERIC(6,4),
    recall_weighted  NUMERIC(6,4),
    hyperparameters JSONB,
    model_path      TEXT,
    notes           TEXT
);

-- ---------------------------------------------------------------------------
-- Tabela de predicoes (opcional - para armazenar resultados de inferencia)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS well_events.predictions (
    id              BIGSERIAL    PRIMARY KEY,
    well_id         VARCHAR(10),
    timestamp_event TIMESTAMP,
    classe_real     INTEGER,
    classe_predita  INTEGER,
    probabilidade   NUMERIC(6,4),
    modelo_versao   VARCHAR(50),
    predicted_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (classe_real)   REFERENCES well_events.event_types(classe),
    FOREIGN KEY (classe_predita) REFERENCES well_events.event_types(classe)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_predictions_well_id
    ON well_events.predictions (well_id);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON well_events.predictions (timestamp_event);

CREATE INDEX IF NOT EXISTS idx_model_metrics_experiment
    ON well_events.model_metrics (experiment_name);

-- ---------------------------------------------------------------------------
-- Grant para o usuario da aplicacao
-- ---------------------------------------------------------------------------

GRANT ALL PRIVILEGES ON SCHEMA well_events TO glue_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA well_events TO glue_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA well_events TO glue_user;

-- ---------------------------------------------------------------------------
-- Dados iniciais de pocos para desenvolvimento
-- ---------------------------------------------------------------------------

INSERT INTO well_events.wells (well_id, well_name) VALUES
    ('00002', 'Well 00002 (Dataset 3W)'),
    ('00004', 'Well 00004 (Dataset 3W)'),
    ('00006', 'Well 00006 (Dataset 3W)')
ON CONFLICT (well_id) DO NOTHING;

\echo 'Banco de dados inicializado com sucesso.'
