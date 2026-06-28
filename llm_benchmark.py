#!/usr/bin/env python3
"""LLM Benchmark - Évalue un modèle via MMLU et retourne score + tokens."""

import os
import sys
import time
import random
import requests
from dotenv import load_dotenv

# Charger la configuration
load_dotenv()

API_KEY = os.getenv("OPENCODE_API_KEY")
MODEL = os.getenv("MODEL", "mimo-v2.5")
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "10"))
API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
TIMEOUT = 60
RATE_LIMIT_DELAY = 1.0  # secondes entre requêtes


CACHE_FILE = os.path.join(os.path.dirname(__file__), "codemmlu_cache.json")
TARGET_TOKENS = 1000


def pad_prompt_to_tokens(prompt: str, target: int = TARGET_TOKENS) -> str:
    """Tronque ou complete le prompt pour atteindre exactement `target` tokens."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt)

    if len(tokens) > target:
        # Tronquer
        tokens = tokens[:target]
    elif len(tokens) < target:
        # Ajouter un séparateur '0' pour éviter la fusion des tokens
        separator = enc.encode("0")
        tokens = tokens + separator
        needed = target - len(tokens)
        if needed > 0:
            # 'a\n' = 2 tokens stables par paire
            pairs = needed // 2
            remainder = needed % 2
            padding = list(enc.encode("a\n")) * pairs
            if remainder:
                padding.append(enc.encode("a")[0])
            tokens = tokens + padding

    return enc.decode(tokens)


def load_mmlu(num_questions: int, seed: int = 42) -> list[dict]:
    """Charge et échantillonne le dataset CodeMMLU depuis HuggingFace (avec cache local)."""
    import json

    # Charger depuis le cache si disponible
    if os.path.exists(CACHE_FILE):
        print("Chargement depuis le cache local...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
    else:
        print("Téléchargement du dataset CodeMMLU depuis HuggingFace...")
        from datasets import load_dataset, concatenate_datasets

        # Sujets CodeMMLU
        subsets = [
            "programming_syntax", "api_frameworks", "software_principles",
            "dbms_sql", "others", "code_completion", "fill_in_the_middle",
            "code_repair", "execution_prediction",
        ]

        datasets_list = []
        for subset in subsets:
            try:
                ds = load_dataset("Fsoft-AIC/CodeMMLU", subset, split="test")
                datasets_list.append(ds)
                print(f"  + {subset}: {len(ds)} questions")
            except Exception:
                try:
                    ds = load_dataset("Fsoft-AIC/CodeMMLU", subset, split="validation")
                    datasets_list.append(ds)
                    print(f"  + {subset}: {len(ds)} questions (validation)")
                except Exception as e:
                    print(f"  ! {subset}: ignoré ({e})")

        dataset = concatenate_datasets(datasets_list) if datasets_list else datasets_list[0]

        all_questions = []
        for item in dataset:
            # CodeMMLU a les colonnes: question, choices, answer, subject/category
            choices = item.get("choices") or item.get("options") or []
            answer_raw = item.get("answer") or item.get("correct") or 0

            # Normaliser la réponse en index 0-3
            if isinstance(answer_raw, str):
                answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer = answer_map.get(answer_raw.upper(), 0)
            else:
                answer = int(answer_raw)

            all_questions.append({
                "question": item.get("question", ""),
                "choices": choices if isinstance(choices, list) else [str(choices)],
                "answer": answer,
                "subject": item.get("subject") or item.get("category") or "programming",
            })

        # Sauvegarder en cache
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False)
        print(f"  Cache sauvegardé: {len(all_questions)} questions")

    # Échantillonner aléatoirement avec seed pour reproductibilité
    rng = random.Random(seed)
    if num_questions < len(all_questions):
        indices = rng.sample(range(len(all_questions)), num_questions)
        questions = [all_questions[i] for i in indices]
    else:
        questions = all_questions

    print(f"  {len(questions)} questions depuis {len(all_questions)} total")
    return questions


def call_opencode(question: dict, max_retries: int = 2) -> dict:
    """Interroge le modèle via l'API OpenCode Go. Retourne réponse + tokens."""
    choices_text = "\n".join([
        f"{'ABCD'[i]}. {c}" for i, c in enumerate(question["choices"])
    ])
    prompt = f"""{question['question']}

{choices_text}"""

    # Pad/truncate à exactement 1000 tokens
    prompt = pad_prompt_to_tokens(prompt, TARGET_TOKENS)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"].get("content") or ""
            content = content.strip().upper()
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)

            model_answer = None
            for char in content:
                if char in "ABCD":
                    model_answer = char
                    break

            return {
                "model_answer": model_answer,
                "total_tokens": total_tokens,
                "raw_response": content,
                "error": None,
            }

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, attente {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {
                "model_answer": None,
                "total_tokens": 0,
                "raw_response": "",
                "error": str(e),
            }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {
                "model_answer": None,
                "total_tokens": 0,
                "raw_response": "",
                "error": str(e),
            }

    return {
        "model_answer": None,
        "total_tokens": 0,
        "raw_response": "",
        "error": "Max retries exceeded",
    }


def run_benchmark(questions: list[dict]) -> dict:
    """Exécute le benchmark sur toutes les questions."""
    correct = 0
    total_tokens = 0
    results = []

    for i, q in enumerate(questions):
        print(f"  Q{i+1}/{len(questions)}: {q['subject'][:30]}...", end=" ")

        result = call_opencode(q)
        tokens = result["total_tokens"]
        total_tokens += tokens

        answer_labels = ["A", "B", "C", "D"]
        correct_answer = answer_labels[q["answer"]]
        is_correct = result["model_answer"] == correct_answer

        if is_correct:
            correct += 1
            print(f"✓ (tokens: {tokens})")
        else:
            print(f"✗ (got {result['model_answer']}, expected {correct_answer}, tokens: {tokens})")

        results.append({
            "subject": q["subject"],
            "correct": is_correct,
            "model_answer": result["model_answer"],
            "expected": correct_answer,
            "tokens": tokens,
            "error": result["error"],
        })

        # Rate limiting
        if i < len(questions) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    accuracy = correct / len(questions) if questions else 0
    score = (accuracy * 10000) / total_tokens if total_tokens > 0 else 0

    return {
        "correct": correct,
        "total": len(questions),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "score": score,
        "results": results,
    }


def main():
    """Point d'entrée principal."""
    if not API_KEY:
        print("Erreur: OPENCODE_API_KEY manquant dans .env")
        sys.exit(1)

    print(f"=== LLM Benchmark ===")
    print(f"Model: {MODEL}")
    print(f"Questions: {NUM_QUESTIONS}")
    print()

    questions = load_mmlu(NUM_QUESTIONS)
    print()

    print("Exécution du benchmark...")
    report = run_benchmark(questions)
    print()

    # Afficher les résultats
    print(f"Accuracy: {report['accuracy']*100:.1f}% ({report['correct']}/{report['total']})")
    print(f"Total tokens: {report['total_tokens']:,}")
    print(f"Score: {report['score']:.2f}")


if __name__ == "__main__":
    main()
