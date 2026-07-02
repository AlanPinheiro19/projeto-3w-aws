#!/usr/bin/env python3
"""
populate_project_board.py
=========================
Popula o GitHub Project Board com todas as tarefas do projeto 3W Petrobras.

Uso:
    export GITHUB_TOKEN=ghp_seu_token
    python scripts/populate_project_board.py --project 1

    # Listar projetos disponíveis:
    python scripts/populate_project_board.py --list

Requisitos:
    pip install requests
"""

import os
import sys
import time
import argparse
import requests

# ── Configuração ────────────────────────────────────────────────────────────
OWNER  = "AlanPinheiro19"
GRAPHQL = "https://api.github.com/graphql"

TOKEN = os.environ.get("GITHUB_TOKEN", "")
if not TOKEN:
    sys.exit("Erro: defina GITHUB_TOKEN antes de rodar.\n"
             "  export GITHUB_TOKEN=ghp_seu_token")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# ── Tarefas do projeto ───────────────────────────────────────────────────────
# Formato: (título, status)
# Status permitidos: "Done", "In Progress", "Todo"
# Ajuste os nomes exatamente como aparecem nas opções do seu board.

TASKS = [
    # ── DONE ──────────────────────────────────────────────────────────────
    ("Configurar repositório GitHub e estrutura do projeto",              "Done"),
    ("Ingestão unificada de dados 3W Petrobras (3 poços, 6 classes)",     "Done"),
    ("ETL Bronze: schema padronizado + campo bronze_at",                  "Done"),
    ("ETL Silver: validação física + fill NaN + deduplicação",            "Done"),
    ("ETL Gold: 136 features (rolling, lag, delta) + Z-Score + split 70/15/15", "Done"),
    ("Pipeline executado na nuvem: 7,48M registros processados",          "Done"),
    ("Treinamento Random Forest baseline (F1-macro 91,1%)",               "Done"),
    ("Treinamento XGBoost baseline (F1-macro 90,0%)",                     "Done"),
    ("Treinamento LightGBM baseline (F1-macro 96,2%) ★ melhor modelo",   "Done"),
    ("Threshold Tuning: RF→93,7% | XGB→95,7% | LGB→96,2%",              "Done"),
    ("SHAP Analysis — interpretabilidade do LightGBM",                   "Done"),
    ("Fix OOM EC2: iter_batches PyArrow + execução sequencial ML",        "Done"),
    ("Deploy AWS Academy: EC2 t3.large + Docker Compose stack",           "Done"),
    ("Configurar swap 4GB no EC2 para treino ML sem OOM",                 "Done"),
    ("Dashboard Streamlit: 6 abas + demo de predição interativa",        "Done"),
    ("Monografia TCC v2 — Capítulos 1 a 5 (ABNT NBR 14724)",             "Done"),
    ("Cronograma do projeto atualizado (jun/2026)",                       "Done"),
    ("PR #2: docs/resultados-ml-nuvem → merged",                         "Done"),
    ("PR #4: docs/aws-deploy-guide → merged",                            "Done"),
    ("GitHub Achievement: YOLO conquistado",                              "Done"),
    ("GitHub Achievement: Quickdraw conquistado",                         "Done"),
    ("Sync dados Gold e modelos para S3 (aws s3 sync)",                  "Done"),
    ("Terraform: módulos EC2, S3, IAM, VPC provisionados",               "Done"),
    ("README.md atualizado com infraestrutura AWS e badges",             "Done"),
    ("AWS_DEPLOY.md: guia completo de deploy na nuvem",                   "Done"),

    # ── IN PROGRESS ───────────────────────────────────────────────────────
    ("Pull Shark Bronze — aguardando processamento GitHub (2 PRs merged)", "In Progress"),
    ("Revisão final monografia: formatação ABNT + referências",           "In Progress"),

    # ── TODO ──────────────────────────────────────────────────────────────
    ("Slides para defesa (15–20 slides, PowerPoint)",                    "Todo"),
    ("Capítulo 5 — Conclusão: revisão e refinamento final",              "Todo"),
    ("GitHub Achievement: Pair Extraordinaire (necessita colaborador)",   "Todo"),
    ("Ensaio de apresentação oral do TCC",                               "Todo"),
]

# ── Helpers GraphQL ──────────────────────────────────────────────────────────

def gql(query: str, variables: dict = None, allow_partial: bool = False) -> dict:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(GRAPHQL, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        if allow_partial and "data" in data and data["data"]:
            # Erros parciais (ex: alguns nodes FORBIDDEN) — continua com o que veio
            pass
        else:
            raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data.get("data", {})


def list_projects() -> list[dict]:
    q = """
    query($owner: String!) {
      user(login: $owner) {
        projectsV2(first: 20) {
          nodes { number title id }
        }
      }
    }
    """
    data = gql(q, {"owner": OWNER}, allow_partial=True)
    nodes = data.get("user", {}).get("projectsV2", {}).get("nodes", [])
    # Filtra nodes None (os que retornaram FORBIDDEN)
    return [n for n in nodes if n]


def get_project(number: int) -> dict:
    q = """
    query($owner: String!, $number: Int!) {
      user(login: $owner) {
        projectV2(number: $number) {
          id
          title
          fields(first: 30) {
            nodes {
              ... on ProjectV2SingleSelectField {
                id
                name
                options { id name }
              }
            }
          }
        }
      }
    }
    """
    data = gql(q, {"owner": OWNER, "number": number}, allow_partial=True)
    return data["user"]["projectV2"]


def get_existing_items(project_id: str) -> dict:
    """Retorna dict {título: item_id} de todos os itens já no projeto."""
    q = """
    query($projectId: ID!, $cursor: String) {
      node(id: $projectId) {
        ... on ProjectV2 {
          items(first: 100, after: $cursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              content {
                ... on DraftIssue { title }
                ... on Issue      { title }
                ... on PullRequest { title }
              }
            }
          }
        }
      }
    }
    """
    existing = {}
    cursor = None
    while True:
        data = gql(q, {"projectId": project_id, "cursor": cursor}, allow_partial=True)
        items = data["node"]["items"]
        for node in items["nodes"]:
            if node and node.get("content"):
                title = node["content"].get("title", "")
                if title:
                    existing[title] = node["id"]
        if not items["pageInfo"]["hasNextPage"]:
            break
        cursor = items["pageInfo"]["endCursor"]
    return existing


def add_draft_item(project_id: str, title: str) -> str:
    m = """
    mutation($projectId: ID!, $title: String!) {
      addProjectV2DraftIssue(input: {projectId: $projectId, title: $title}) {
        projectItem { id }
      }
    }
    """
    data = gql(m, {"projectId": project_id, "title": title})
    return data["addProjectV2DraftIssue"]["projectItem"]["id"]


def set_status(project_id: str, item_id: str, field_id: str, option_id: str):
    m = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $fieldId,
        value: { singleSelectOptionId: $optionId }
      }) {
        projectV2Item { id }
      }
    }
    """
    gql(m, {
        "projectId": project_id,
        "itemId":    item_id,
        "fieldId":   field_id,
        "optionId":  option_id,
    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Popula GitHub Project Board — projeto 3W")
    parser.add_argument("--project", type=int, default=None,
                        help="Número do projeto (ex: 1)")
    parser.add_argument("--list",    action="store_true",
                        help="Lista projetos disponíveis e sai")
    args = parser.parse_args()

    # Listar projetos
    projects = list_projects()
    if args.list or not projects:
        print(f"\nProjetos de @{OWNER}:")
        for p in projects:
            print(f"  #{p['number']}  {p['title']}")
        if not projects:
            print("  Nenhum projeto encontrado. Crie um em github.com/AlanPinheiro19?tab=projects")
        return

    # Escolher projeto
    project_number = args.project
    if project_number is None:
        if len(projects) == 1:
            project_number = projects[0]["number"]
            print(f"Usando único projeto encontrado: #{project_number} — {projects[0]['title']}")
        else:
            print("\nProjetos disponíveis:")
            for p in projects:
                print(f"  #{p['number']}  {p['title']}")
            project_number = int(input("\nDigite o número do projeto: "))

    # Carregar projeto
    project = get_project(project_number)
    print(f"\nProjeto: {project['title']}")
    project_id = project["id"]

    # Encontrar campo Status e suas opções
    status_field = None
    for field in project["fields"]["nodes"]:
        if field and field.get("name", "").lower() == "status":
            status_field = field
            break

    if not status_field:
        print("\nERRO: Campo 'Status' não encontrado no projeto.")
        print("Campos disponíveis:")
        for f in project["fields"]["nodes"]:
            if f:
                print(f"  - {f.get('name','?')}")
        print("\nCrie o campo Status no projeto antes de rodar este script.")
        sys.exit(1)

    field_id = status_field["id"]
    options   = {opt["name"]: opt["id"] for opt in status_field["options"]}
    print(f"Campo Status encontrado. Opções: {list(options.keys())}")

    # Mapear status → opção do board (ajusta se necessário)
    STATUS_MAP = {}
    for task_status in ["Done", "In Progress", "Todo"]:
        # Tenta match exato, depois parcial
        matched = options.get(task_status)
        if not matched:
            for opt_name, opt_id in options.items():
                if task_status.lower() in opt_name.lower():
                    matched = opt_id
                    break
        if matched:
            STATUS_MAP[task_status] = matched
        else:
            print(f"AVISO: opção '{task_status}' não encontrada — itens com esse status serão adicionados sem status.")

    # Buscar itens já existentes (evita duplicatas)
    print("\nVerificando itens existentes no board...")
    existing = get_existing_items(project_id)
    print(f"  {len(existing)} itens encontrados.")

    # Processar tarefas: criar se não existe, só atualizar status se já existe
    print(f"\nSincronizando {len(TASKS)} tarefas...\n")
    total = len(TASKS)
    criados = 0
    atualizados = 0
    erros = 0

    for i, (title, status) in enumerate(TASKS, 1):
        try:
            option_id = STATUS_MAP.get(status)

            if title in existing:
                # Já existe — só atualiza o status
                item_id = existing[title]
                if option_id:
                    set_status(project_id, item_id, field_id, option_id)
                action = "atualizado"
                atualizados += 1
            else:
                # Não existe — cria e define status
                item_id = add_draft_item(project_id, title)
                if option_id:
                    set_status(project_id, item_id, field_id, option_id)
                action = "criado   "
                criados += 1

            label = f"[{status}]"
            print(f"  {i:2}/{total}  {label:15}  {action}  {title[:60]}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {i:2}/{total}  ERRO: {title[:50]} — {e}")
            erros += 1

    print(f"\nConcluído! Criados: {criados} | Atualizados: {atualizados} | Erros: {erros}")

    print(f"\nConcluído! Acesse: https://github.com/users/{OWNER}/projects/{project_number}")


if __name__ == "__main__":
    main()
