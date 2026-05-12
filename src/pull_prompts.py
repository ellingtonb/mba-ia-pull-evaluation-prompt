"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""
import datetime
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from utils import save_yaml, check_env_vars, print_section_header


load_dotenv()

required_env_vars = ['LANGSMITH_ENDPOINT', 'LANGSMITH_API_KEY', 'LANGSMITH_TRACING']

prompt_name = os.getenv("LANGSMITH_HUB_PROMPT", "leonanluppi/bug_to_user_story_v1")

default_prompt_description = "Prompt para converter relatos de bugs em User Stories"


def pull_prompts_from_langsmith():
    client = Client()
    prompt = client.pull_prompt(
        prompt_identifier=prompt_name,
        dangerously_pull_public_prompt=True
    )

    print("Prompt carregado com sucesso do LangSmith Hub!")

    return prompt


def convert_prompt_to_yaml_format(prompt):
    prompt_key = prompt_name.split("/")[-1]
    prompt_version = prompt_name.split("_v")[-1]

    system_prompt = ""
    user_prompt = ""

    for message in prompt.messages:
        message_type = message.__class__.__name__
        template = message.prompt.template

        if message_type == "SystemMessagePromptTemplate":
            system_prompt = template
        elif message_type == "HumanMessagePromptTemplate":
            user_prompt = template

    tags = [
        f"{key}:{value}"
        for key, value in prompt.metadata.items()
    ]
    tags.append(f"version:{prompt_version}")

    return {
        prompt_key: {
            "description": default_prompt_description,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": f"v{prompt_version}",
            "created_at": datetime.date.today().strftime("%Y-%m-%d"),
            "tags": tags
        }
    }


def main():
    try:
        print_section_header(f"Prompt: {prompt_name}")

        if not check_env_vars(required_env_vars):
            return 1

        if os.getenv('LANGSMITH_TRACING') == 'false':
            print(f"\n❌ Habilite o tracing do LangSmith para verificar o que a LLM está fazendo.")
            return 1

        output_path = f"{Path(__file__).parent.parent}/prompts/{prompt_name.split('/')[-1]}.yml"

        if Path(output_path).exists():
            user_input = input("Deseja substituir o prompt existente? (s/n): ").lower()
            if user_input == 's':
                os.remove(output_path)
                print("")
            else:
                print("\n⚠️ Download cancelado!")
                return 0

        prompt = pull_prompts_from_langsmith()
        yaml_prompt = convert_prompt_to_yaml_format(prompt)

        save_yaml(yaml_prompt, output_path)
        print(f"\n✅ Prompt salvo localmente com sucesso!")

        return 0
    except Exception as e:
        print(f"\n❌ Erro ao carregar o prompt: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
