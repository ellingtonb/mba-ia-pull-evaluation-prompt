"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 5 métricas (Helpfulness, Correctness, F1-Score, Clarity, Precision)
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision
from langsmith.evaluation import evaluate as langsmith_evaluate
from langsmith.schemas import Example, Run

load_dotenv()


def get_llm():
    return get_configured_llm(temperature=0)


def _create_predict_fn(prompt_name: str, client: Client):
    prompt_template = pull_prompt_from_langsmith(prompt_name, client)
    llm = get_llm()
    def predict(inputs: dict) -> dict:
        chain = prompt_template | llm
        response = chain.invoke(inputs)
        return {"output": response.content}
    return predict


def _create_langsmith_evaluators():

    def combined_evaluator(run: Run, example: Example) -> list:
        answer = run.outputs.get("output", "") if run.outputs else ""
        reference = example.outputs.get("reference", "") if example.outputs else ""
        inputs = example.inputs or {}
        question = inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A")))

        f1 = evaluate_f1_score(question, answer, reference)
        clarity = evaluate_clarity(question, answer, reference)
        precision = evaluate_precision(question, answer, reference)

        helpfulness = (clarity["score"] + precision["score"]) / 2
        correctness = (f1["score"] + precision["score"]) / 2

        return [
            {"key": "f1_score", "score": f1["score"], "comment": f1.get("reasoning", "")},
            {"key": "clarity", "score": clarity["score"], "comment": clarity.get("reasoning", "")},
            {"key": "precision", "score": precision["score"], "comment": precision.get("reasoning", "")},
            {"key": "helpfulness", "score": round(helpfulness, 4)},
            {"key": "correctness", "score": round(correctness, 4)},
        ]

    return [combined_evaluator]


def _ensure_evaluator_configs(client: Client):
    metric_keys = ["f1_score", "clarity", "precision", "helpfulness", "correctness"]
    config = {"type": "continuous", "min": 0.0, "max": 1.0}
    existing_cfgs = list(client.list_feedback_configs())
    existing_keys = {cfg.feedback_key for cfg in existing_cfgs}
    for key in metric_keys:
        if key not in existing_keys:
            client.create_feedback_config(feedback_key=key, feedback_config=config)
            print(f"   ✓ Evaluator config criado: {key}")
        else:
            cfg = next(c for c in existing_cfgs if c.feedback_key == key)
            stored = cfg.feedback_config
            if stored.get("min") != 0.0 or stored.get("max") != 1.0:
                client.update_feedback_config(key, feedback_config=config)
                print(f"   ✓ Evaluator config atualizado: {key}")


def _evaluate_prompt_langsmith(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando com LangSmith Experiments: {prompt_name}")
    print(f"   Dataset: {dataset_name}")

    try:
        _ensure_evaluator_configs(client)

        predict_fn = _create_predict_fn(prompt_name, client)
        evaluators = _create_langsmith_evaluators()

        print("   Executando experimento e avaliação no LangSmith...\n")

        results = langsmith_evaluate(
            predict_fn,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=f"eval-{prompt_name.replace('/', '-').replace(':', '-')}",
            client=client,
            blocking=True,
            upload_results=True,
        )

        experiment_name = results.experiment_name
        print(f"\n   ✓ Experimento criado: {experiment_name}")

        f1_scores = []
        clarity_scores = []
        precision_scores = []
        helpfulness_scores = []
        correctness_scores = []

        print("   Coletando resultados...")

        for example_idx, row in enumerate(results, 1):
            row_scores = {}
            row_comments = {}
            eval_results = row.get("evaluation_results", {})
            eval_list = eval_results.get("results", []) if isinstance(eval_results, dict) else []

            for eval_result in eval_list:
                key = eval_result.key if hasattr(eval_result, 'key') else eval_result.get("key", "")
                score = eval_result.score if hasattr(eval_result, 'score') else eval_result.get("score", 0.0)
                comment = eval_result.comment if hasattr(eval_result, 'comment') else eval_result.get("comment", "")
                row_scores[key] = score
                row_comments[key] = comment

            f1_scores.append(row_scores.get("f1_score", 0.0))
            clarity_scores.append(row_scores.get("clarity", 0.0))
            precision_scores.append(row_scores.get("precision", 0.0))
            helpfulness_scores.append(row_scores.get("helpfulness", 0.0))
            correctness_scores.append(row_scores.get("correctness", 0.0))

            print(f"      [{example_idx}] F1:{row_scores.get('f1_score', 0):.2f} "
                  f"Clarity:{row_scores.get('clarity', 0):.2f} "
                  f"Precision:{row_scores.get('precision', 0):.2f} "
                  f"Helpfulness:{row_scores.get('helpfulness', 0):.2f} "
                  f"Correctness:{row_scores.get('correctness', 0):.2f}")

            debug_low_scores = os.getenv("DEBUG_LOW_SCORES", "false").lower() == "true"
            debug_threshold = float(os.getenv("DEBUG_SCORE_THRESHOLD", "0.90"))
            has_low_score = (
                    row_scores.get('f1_score', 0) < debug_threshold
                    or row_scores.get('clarity', 0) < debug_threshold
                    or row_scores.get('precision', 0) < debug_threshold
            )

            if debug_low_scores and has_low_score:
                run = row.get("run")
                example = row.get("example")
                inputs = (example.inputs or {}) if example else ((run.inputs or {}) if run else {})

                result = {
                    "question": inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A"))),
                    "answer": run.outputs.get("output", "") if run and run.outputs else "",
                    "reference": example.outputs.get("reference", "") if example and example.outputs else ""
                }
                f1 = {"score": row_scores.get("f1_score", 0), "reasoning": row_comments.get("f1_score", "")}
                clarity = {"score": row_scores.get("clarity", 0), "reasoning": row_comments.get("clarity", "")}
                precision = {"score": row_scores.get("precision", 0), "reasoning": row_comments.get("precision", "")}

                print("\n" + "-" * 70)
                print(f"DEBUG EXEMPLO {example_idx}")
                print("-" * 70)
                print("\nBUG REPORT:")
                print(result["question"])
                print("\nRESPOSTA GERADA:")
                print(result["answer"])
                print("\nREFERÊNCIA:")
                print(result["reference"])
                print("\nF1 REASONING:")
                print(f1.get("reasoning", ""))
                print("\nCLARITY REASONING:")
                print(clarity.get("reasoning", ""))
                print("\nPRECISION REASONING:")
                print(precision.get("reasoning", ""))
                print("-" * 70 + "\n")

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_helpfulness = sum(helpfulness_scores) / len(helpfulness_scores) if helpfulness_scores else 0.0
        avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0

        return {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4)
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação LangSmith: {e}")
        import traceback
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            "helpfulness": 0.0,
            "correctness": 0.0,
            "f1_score": 0.0,
            "clarity": 0.0,
            "precision": 0.0
        }


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"   ✓ Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"   ✓ Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )

            print(f"   ✓ Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str, client: Client) -> ChatPromptTemplate:
    try:
        print(f"   Puxando prompt do LangSmith Hub: {prompt_name}")
        prompt = client.pull_prompt(
            prompt_identifier = prompt_name,
            dangerously_pull_public_prompt = True
        )
        print(f"   ✓ Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("⚠️  O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'")
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, 'inputs') else {}
        outputs = example.outputs if hasattr(example, 'outputs') else {}

        chain = prompt_template | llm

        response = chain.invoke(inputs)
        answer = response.content

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        if isinstance(inputs, dict):
            question = inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A")))
        else:
            question = "N/A"

        return {
            "answer": answer,
            "reference": reference,
            "question": question
        }

    except Exception as e:
        print(f"      ⚠️  Erro ao avaliar exemplo: {e}")
        import traceback
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            "answer": "",
            "reference": "",
            "question": ""
        }


def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    if os.getenv("LANGSMITH_EVALUATION_ENABLED", "false").lower() == "true":
        return _evaluate_prompt_langsmith(prompt_name, dataset_name, client)

    print(f"\n🔍 Avaliando: {prompt_name}")

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name, client)

        examples = list(client.list_examples(dataset_name=dataset_name))
        print(f"   Dataset: {len(examples)} exemplos")

        llm = get_llm()

        f1_scores = []
        clarity_scores = []
        precision_scores = []

        print("   Avaliando exemplos...")

        for i, example in enumerate(examples, 1):
            result = evaluate_prompt_on_example(prompt_template, example, llm)

            if result["answer"]:
                f1 = evaluate_f1_score(result["question"], result["answer"], result["reference"])
                clarity = evaluate_clarity(result["question"], result["answer"], result["reference"])
                precision = evaluate_precision(result["question"], result["answer"], result["reference"])

                f1_scores.append(f1["score"])
                clarity_scores.append(clarity["score"])
                precision_scores.append(precision["score"])

                print(f"      [{i}/{len(examples)}] F1:{f1['score']:.2f} Clarity:{clarity['score']:.2f} Precision:{precision['score']:.2f}")

                debug_low_scores = os.getenv("DEBUG_LOW_SCORES", "false").lower() == "true"
                debug_threshold = float(os.getenv("DEBUG_SCORE_THRESHOLD", "0.90"))
                has_low_score = (
                    f1["score"] < debug_threshold
                    or clarity["score"] < debug_threshold
                    or precision["score"] < debug_threshold
                )

                if debug_low_scores and has_low_score:
                    print("\n" + "-" * 70)
                    print(f"DEBUG EXEMPLO {i}")
                    print("-" * 70)
                    print("\nBUG REPORT:")
                    print(result["question"])
                    print("\nRESPOSTA GERADA:")
                    print(result["answer"])
                    print("\nREFERÊNCIA:")
                    print(result["reference"])
                    print("\nF1 REASONING:")
                    print(f1.get("reasoning", ""))
                    print("\nCLARITY REASONING:")
                    print(clarity.get("reasoning", ""))
                    print("\nPRECISION REASONING:")
                    print(precision.get("reasoning", ""))
                    print("-" * 70 + "\n")

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

        avg_helpfulness = (avg_clarity + avg_precision) / 2
        avg_correctness = (avg_f1 + avg_precision) / 2

        return {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4)
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação: {e}")
        return {
            "helpfulness": 0.0,
            "correctness": 0.0,
            "f1_score": 0.0,
            "clarity": 0.0,
            "precision": 0.0
        }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Derivadas:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=0.9)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=0.9)}")

    print("\nMétricas Base:")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=0.9)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=0.9)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=0.9)}")

    average_score = sum(scores.values()) / len(scores)

    print("\n" + "-" * 50)
    print(f"📊 MÉDIA GERAL: {average_score:.4f}")
    print("-" * 50)

    all_above_threshold = all(score >= 0.9 for score in scores.values())
    passed = all_above_threshold and average_score >= 0.9

    if passed:
        print(f"\n✅ STATUS: APROVADO - Todas as métricas >= 0.9")
    else:
        print(f"\n❌ STATUS: REPROVADO")
        failed_metrics = [name for name, score in scores.items() if score < 0.9]
        if failed_metrics:
            print(f"⚠️  Métricas abaixo de 0.9: {', '.join(failed_metrics)}")
        print(f"⚠️  Média atual: {average_score:.4f} | Necessário: 0.9000")

    return passed


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}")

    debug_low_scores = os.getenv("DEBUG_LOW_SCORES", "false").lower() == "true"
    print(f"Debug Low Scores: {debug_low_scores}")
    debug_threshold = float(os.getenv("DEBUG_SCORE_THRESHOLD", "0.90"))
    if debug_low_scores:
        print(f"Debug Score Threshold: {debug_threshold}")

    eval_enabled = os.getenv("LANGSMITH_EVALUATION_ENABLED", "false").lower() == "true"
    if eval_enabled:
        print("📡 LangSmith Evaluation ativado: Experiments e Evaluations serão enviados para o LangSmith\n")
    else:
        print()

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-resolved")

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    if not username:
        print("❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        print("   Configure seu username do LangSmith Hub antes de continuar.")
        return 1

    prompts_to_evaluate = [
        f"{username}/bug_to_user_story_v2",
    ]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {
                    "helpfulness": 0.0,
                    "correctness": 0.0,
                    "f1_score": 0.0,
                    "clarity": 0.0,
                    "precision": 0.0
                },
                "passed": False
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("⚠️  Nenhum prompt foi avaliado")
        return 1

    print(f"Prompts avaliados: {evaluated_count}")
    print(f"Aprovados: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print("✅ Todos os prompts atingiram todas as métricas >= 0.9!")
        print(f"\n✓ Confira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("⚠️  Alguns prompts não atingiram todas as métricas >= 0.9")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1

if __name__ == "__main__":
    sys.exit(main())
