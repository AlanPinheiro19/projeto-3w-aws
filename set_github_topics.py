"""
set_github_topics.py
--------------------
Adiciona as tags (topics) ao repositório GitHub do TCC 3W Petrobras.

USO:
    # Defina o token antes de rodar:
    set GITHUB_TOKEN=ghp_SEU_TOKEN_AQUI   (CMD)
    $env:GITHUB_TOKEN="ghp_SEU_TOKEN"     (PowerShell)

    python set_github_topics.py

ESCOPO NECESSÁRIO:
  - Repositório público  → escopo 'public_repo'
  - Repositório privado  → escopo 'repo' (completo)
Gere em: https://github.com/settings/tokens
"""

import os
import sys
import requests

# ── Configuração ──────────────────────────────────────────────────────────────
OWNER = "AlanPinheiro19"
REPO  = "projeto-3w-aws"

# Topics que refletem as principais tecnologias e habilidades do projeto
# GitHub aceita no máximo 20 topics por repositório
TOPICS = [
    # Linguagem e plataforma
    "python",
    "docker",
    # Machine Learning
    "machine-learning",
    "lightgbm",
    "xgboost",
    "random-forest",
    "shap",
    "optuna",
    # Engenharia de Dados
    "data-engineering",
    "etl-pipeline",
    "medallion-architecture",
    "apache-airflow",
    "pyspark",
    "aws-glue",
    # Infraestrutura
    "minio",
    "postgresql",
    # Domínio
    "time-series",
    "anomaly-detection",
    "petrobras-3w",
    "offshore-oil-wells",
]

# ── Execução ──────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN não encontrado no ambiente.")
        print("   Defina antes de rodar:")
        print('   CMD:        set GITHUB_TOKEN=ghp_...')
        print('   PowerShell: $env:GITHUB_TOKEN="ghp_..."')
        sys.exit(1)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # ── Diagnóstico: verificar usuário autenticado ────────────────────────────
    me = requests.get("https://api.github.com/user", headers=headers)
    if me.status_code != 200:
        print(f"❌ Token inválido (status {me.status_code}). Gere um novo em:")
        print("   https://github.com/settings/tokens")
        sys.exit(1)
    login = me.json().get("login", "?")
    print(f"✅ Autenticado como: {login}")

    # ── Diagnóstico: verificar se o repositório existe ────────────────────────
    repo_url = f"https://api.github.com/repos/{OWNER}/{REPO}"
    repo_resp = requests.get(repo_url, headers=headers)

    if repo_resp.status_code == 404:
        # Listar repositórios do usuário para ajudar a identificar o nome certo
        print(f"\n❌ Repositório '{OWNER}/{REPO}' não encontrado.")
        print("   Listando seus repositórios para identificar o nome correto:\n")
        repos_resp = requests.get(
            f"https://api.github.com/users/{login}/repos?per_page=50&sort=updated",
            headers=headers
        )
        if repos_resp.status_code == 200:
            for r in repos_resp.json():
                vis = "🔒 privado" if r.get("private") else "🌐 público"
                print(f"  {vis}  {r['full_name']}")
        print("\n   Atualize OWNER e REPO no topo deste script e rode novamente.")
        sys.exit(1)
    elif repo_resp.status_code != 200:
        print(f"❌ Erro ao acessar repositório: {repo_resp.status_code} — {repo_resp.text}")
        sys.exit(1)

    repo_data = repo_resp.json()
    visibility = "privado 🔒" if repo_data.get("private") else "público 🌐"
    print(f"✅ Repositório encontrado: {repo_data['full_name']} ({visibility})")

    # ── Aplicar topics ────────────────────────────────────────────────────────
    topics_url = f"https://api.github.com/repos/{OWNER}/{REPO}/topics"
    print(f"\nAplicando {len(TOPICS)} topics...")

    resp = requests.put(topics_url, json={"names": TOPICS}, headers=headers)

    if resp.status_code == 200:
        result = resp.json().get("names", [])
        print(f"\n✅ {len(result)} topics aplicados com sucesso!")
        for t in result:
            print(f"  • {t}")
        print(f"\n   Visualize em: https://github.com/{OWNER}/{REPO}")
    else:
        print(f"❌ Erro {resp.status_code}: {resp.text}")
        sys.exit(1)

if __name__ == "__main__":
    main()
