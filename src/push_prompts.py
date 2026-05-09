"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header, validate_prompt_structure

load_dotenv()


required_env_vars = ['LANGSMITH_ENDPOINT', 'LANGSMITH_API_KEY', 'LANGSMITH_TRACING', 'LANGSMITH_PROJECT', 'USERNAME_LANGSMITH_HUB', 'IMPROVED_PROMPT']

required_tags = ['lc_hub_owner', 'lc_hub_repo', 'version', 'technique', 'parent_prompt', 'parent_prompt_commit_hash']
required_techniques = ['few-shot-learning']

prompt_name = os.getenv("IMPROVED_PROMPT", "bug_to_user_story_v2")


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """

    try:
        hub.push(
            repo_full_name=f"{os.getenv('USERNAME_LANGSMITH_HUB')}/{prompt_name}",
            new_repo_is_public=True,
            new_repo_description=prompt_data.get('description', ''),
            tags=prompt_data.get('tags', []),
            object=ChatPromptTemplate([
                ("system", prompt_data.get('system_prompt', '')),
                ("human", prompt_data.get('user_prompt', ''))
            ])
        )
        print(f"\n✅ Prompt '{os.getenv('USERNAME_LANGSMITH_HUB')}/{prompt_name}' publicado com sucesso!")
        return True
    except Exception as e:
        print(f"\n❌ Erro ao fazer push do prompt '{prompt_name}': {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """

    base_validated_prompt = validate_prompt_structure(prompt_data)
    errors = base_validated_prompt[1]

    tags = prompt_data.get('tags', [])
    if len(tags) == 0:
        errors.append("Prompt deve ter as tags: {}".format(", ".join(required_tags)))
    else:
        techniques_applied = prompt_data.get('techniques_applied', [])
        for technique in techniques_applied:
            if f"technique:{technique}" not in tags:
                errors.append(f"Tag technique:'{technique}' não encontrada na propriedade 'tags'")

        for required_technique in required_techniques:
            if required_technique not in techniques_applied:
                errors.append(f"Técnica '{required_technique}' deve ser uma técnica obrigatória")

        for required_tag in ['parent_prompt', 'parent_prompt_commit_hash']:
            if any(tag.startswith(required_tag) for tag in tags):
                continue
            errors.append(f"Prompt deve ter uma tag '{required_tag}' definida")

        version = prompt_data.get('version', '')
        if not version:
            errors.append("Prompt deve ter uma versão definida na propriedade 'version'")
        if not version.startswith('v'):
            errors.append("Versão do prompt deve começar com 'v' na propriedade 'version'")
        if f"version:{version.split('v')[1]}" not in tags:
            errors.append(f"Tag version:'{version}' não encontrada na propriedade 'tags'")

        if f"lc_hub_owner:{os.getenv('USERNAME_LANGSMITH_HUB')}" not in tags:
            errors.append(f"Tag lc_hub_owner:'{os.getenv('USERNAME_LANGSMITH_HUB')}' não encontrada na propriedade 'tags'")

        if f"lc_hub_repo:{os.getenv('LANGSMITH_PROJECT')}" not in tags:
            errors.append(f"Tag lc_hub_repo:'{os.getenv('LANGSMITH_PROJECT')}' não encontrada na propriedade 'tags'")

    if len(errors) > 0:
        print(f"\n❌ Erro na estrutura do prompt: {errors}")
    else:
        print(f"\n✅ Estrutura do prompt válida!")

    return len(errors) == 0, errors


def main():
    """Função principal"""

    try:
        print_section_header(f"Prompt: {prompt_name}")

        if not check_env_vars(required_env_vars):
            return 1

        if os.getenv('LANGSMITH_TRACING') == 'false':
            print(f"\n❌ Habilite o tracing do LangSmith para verificar o que a LLM está fazendo.")
            return 1

        prompt_path = f"{Path(__file__).parent.parent}/prompts/{prompt_name.split('/')[-1]}.yml"

        if not Path(prompt_path).exists():
            print(f"\n❌ Prompt não encontrado: {prompt_path}")
            return 1

        prompt = load_yaml(prompt_path)

        if not prompt:
            print(f"\n❌ Prompt vazio: {prompt_path}")
            return 1

        prompt_data = prompt.get(prompt_name, {})
        if not prompt_data:
            print(f"\n❌ Propriedade {prompt_name} não encontrada no arquivo {prompt_path}")
            return 1

        validated_prompt = validate_prompt(prompt_data)

        if not validated_prompt[0]:
            print(f"\n❌ Erro na estrutura do prompt:")
            for error in validated_prompt[1]:
                print(f"   - {error}")
            return 1

        push_prompt_to_langsmith(prompt_name, prompt_data)
    except Exception as e:
        print(f"\n❌ Erro ao publicar prompt: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
