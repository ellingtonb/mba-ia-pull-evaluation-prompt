"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
import re
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

default_file_name = 'bug_to_user_story_v2'
default_file_extension = 'yml'

@pytest.fixture(scope="module")
def loaded_prompt():
    """Fixture do Pytest que carrega os prompts do arquivo YAML uma única vez por módulo."""
    file_path = Path(__file__).parent.parent / f"prompts/{default_file_name}.{default_file_extension}"
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TestPrompts:
    def test_prompt_exists_local(self):
        """Verifica se o arquivo de prompts existe localmente."""
        file_path = Path(__file__).parent.parent / f"prompts/{default_file_name}.{default_file_extension}"
        assert file_path.exists(), f"Arquivo de prompts '{file_path}' não encontrado"

    def test_prompt_has_base_property(self, loaded_prompt):
        """Verifica se o campo 'base' existe e não está vazio."""
        assert default_file_name in loaded_prompt, f"Campo '{default_file_name}' não encontrado no prompt"

    def test_prompt_has_system_prompt(self, loaded_prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert 'system_prompt' in loaded_prompt[default_file_name], "Prompt 'system_prompt' não pode estar vazio"

    def test_prompt_has_user_prompt(self, loaded_prompt):
        """Verifica se o campo 'user_prompt' existe e não está vazio."""
        assert 'user_prompt' in loaded_prompt[default_file_name], "Prompt 'user_prompt' não pode estar vazio"

    def test_prompt_has_role_definition(self, loaded_prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = loaded_prompt[default_file_name].get('system_prompt', '')
        assert re.search(r'(?i)você é um(a)?\s+[a-zà-ÿ\s]+', system_prompt), "Persona da Role Definition não encontrada no prompt"

    def test_prompt_mentions_format(self, loaded_prompt):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        required_markdown_mentions = [
            r'^Entregue\s+(?:\w+\s+){0,5}em\s+Markdown$',
            'Use Markdown',
            'usando Markdown'
        ]
        assert any(re.search(pattern, loaded_prompt[default_file_name]['system_prompt'], re.IGNORECASE) for pattern in required_markdown_mentions), "Formato Markdown ou User Story não mencionado no prompt"

        required_user_story_mentions = [
            r'^use\s+(?:\w+\s+){0,5}em\s+User Story$',
            'Apenas User Story',
            'use User Story'
        ]
        assert any(re.search(pattern, loaded_prompt[default_file_name]['system_prompt'], re.IGNORECASE) for pattern in required_user_story_mentions), "Formato User Story não mencionado no prompt"

    def test_prompt_has_few_shot_examples(self, loaded_prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = loaded_prompt[default_file_name].get('system_prompt', '')
        
        # Padrão regex para procurar a estrutura exigida, permitindo variações de acento e quebras de linha
        pattern = r"(?is)exemplo.*?entrada:.*?sa[ií]da:.*?crit[eé]rios de aceita[cç][aã]o:"
        assert re.search(pattern, system_prompt), (
            "O prompt deve conter pelo menos um exemplo estruturado contendo as seções "
            "'Exemplo', 'Entrada:', 'Saída:' e 'Critérios de Aceitação:'"
        )

    def test_prompt_no_todos(self, loaded_prompt):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        assert not re.search(r'\[TODO\]', loaded_prompt[default_file_name]['system_prompt'], re.IGNORECASE), "Encontrado '[TODO]' no prompt"

    def test_minimum_techniques(self, loaded_prompt):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = loaded_prompt[default_file_name].get('techniques_applied', [])
        assert len(techniques) >= 2, "Prompt deve conter pelo menos 2 técnicas aplicadas"

    def test_prompt_has_version(self, loaded_prompt):
        """Verifica se o prompt contém a versão do modelo."""
        assert 'version' in loaded_prompt[default_file_name], "Versão do modelo não mencionada no prompt"
        assert re.search(r'v?\d+(\.\d+)*', loaded_prompt[default_file_name]['version']), "Formato de versão inválido"

    def test_minimum_tags(self, loaded_prompt):
        """Verifica se o prompt contém pelo menos 3 tags relevantes."""
        tags = loaded_prompt[default_file_name].get('tags', [])
        assert len(tags) >= 3, "Prompt deve conter pelo menos 3 tags relevantes"

    def test_prompt_required_tags(self, loaded_prompt):
        """Verifica se o prompt contém tags específicas necessárias."""
        required_tags = {'version', 'technique'}
        tags_list = loaded_prompt[default_file_name].get('tags', [])
        
        prompt_tags = {tag.split(':')[0].strip() for tag in tags_list}

        missing_tags = required_tags - prompt_tags
        assert not missing_tags, f"Tags obrigatórias não encontradas no prompt: {', '.join(missing_tags)}"

    def test_prompt_has_techniques_in_tags(self, loaded_prompt):
        """Verifica se o prompt contém pelo menos uma técnica em cada tag."""
        tags_list = loaded_prompt[default_file_name].get('tags', [])
        techniques = loaded_prompt[default_file_name].get('techniques_applied', [])
        for technique in techniques:
            assert f"technique:{technique}" in tags_list, f"Tag 'technique:{technique}' não encontrada"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
