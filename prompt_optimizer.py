#!/usr/bin/env python3
"""Prompt Optimizer - Itère sur un prompt via benchmark LLM pour maximiser le score."""

import os
import sys
import json
import time
import random
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Charger la config
load_dotenv()

API_KEY = os.getenv("OPENCODE_API_KEY")
MODEL = os.getenv("MODEL", "mimo-v2.5")
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "10"))
WAVES = int(os.getenv("WAVES", "3"))
API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))  # parallel evaluations
SEED = int(os.getenv("SEED", "42"))  # seed for reproducible question selection

# Importer le benchmark
sys.path.insert(0, os.path.dirname(__file__))
from llm_benchmark import load_mmlu, call_opencode, run_benchmark, pad_prompt_to_tokens, TARGET_TOKENS

OPTIMIZER_PROMPT = """CRITICAL RULE: The prompt you generate must NEVER contain answer format instructions. Specifically, these patterns are STRICTLY FORBIDDEN:
- "answer with A, B, C, or D"
- "answer the question correctly by choosing"
- "reply only with the letter"
- "réponds par la lettre"
- "réponds uniquement par"
- Any variation asking the model how to format its answer

The answer extraction is handled automatically by the system. Your prompt must focus ONLY on reasoning strategy, not on telling the model how to respond.

---

Create a way of thinking.

Rules:
- Exactly {target_tokens} tokens (verify with tiktoken cl100k_base)
- Must be a COGNITIVE STRATEGY, not instructions
- Invent a novel reasoning pattern the model hasn't seen
- No domain knowledge — pure thinking innovation
- Return ONLY the thinking pattern, nothing else

{improvement_section}"""


import re


# Patterns interdits dans les prompts générés
FORBIDDEN_PATTERNS = [
    r"(?i)answer\s+(the\s+)?question\s+correctly\s+by\s+choosing",
    r"(?i)answer\s+with\s+",
    r"(?i)reply\s+only\s+with\s+",
    r"(?i)réponds?\s+(par\s+)?la\s+lettre",
    r"(?i)réponds?\s+uniquement\s+par",
    r"(?i)choose\s+from\s+[a-d]",
    r"(?i)select\s+from\s+[a-d]",
    r"(?i)pick\s+the\s+(correct\s+)?(answer|letter|option)",
    r"(?i)^\.?thinking",
    r"(?i)^\.?instruction",
    r"(?i)^\.?system",
    r"(?i)^let\s+me\s+(carefully|parse|think|consider)",
    r"(?i)^we\s+need\s+to\s+generate",
    r"(?i)^the\s+user\s+wants",
    r"(?i)^I\s+need\s+to\s+(create|generate|design)",
    r"(?i)^my\s+task\s+is",
    r"(?i)^goal:",
    r"(?i)^output:\s+only",
]
FORBIDDEN_RE = [re.compile(p) for p in FORBIDDEN_PATTERNS]


def clean_prompt(prompt: str) -> str:
    """Supprime les lignes contenant des instructions de format de réponse interdites."""
    lines = prompt.split("\n")
    cleaned = []
    for line in lines:
        if any(p.search(line) for p in FORBIDDEN_RE):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    # Si le prompt est devenu vide après nettoyage, retourner un fallback
    return result if result else "Decompose this problem into sub-problems. For each, consider what could go wrong. Then synthesize."


def _generate_single_prompt(args: tuple) -> str:
    """Worker pour générer un seul prompt en parallèle."""
    idx, current_prompt, target_tokens = args
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Construire la section d'amélioration
    if current_prompt:
        improvement = f"\n\nImprove this one:\n{current_prompt}"
    else:
        improvement = ""

    prompt_text = OPTIMIZER_PROMPT.format(
        target_tokens=target_tokens,
        improvement_section=improvement,
    )
    prompt_text += f"\n\nGenerate variation #{idx + 1}. You must only respond with your idea."

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a creative prompt designer. Output ONLY the requested text, no explanation."},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.9 + (idx * 0.01),
        "max_tokens": 2000,
    }

    for attempt in range(5):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if len(content) > 10:
                return clean_prompt(content)
            time.sleep(2)
        except Exception:
            time.sleep(3)

    raise RuntimeError(f"Impossible de générer le prompt #{idx} après 5 tentatives")


def generate_prompts(current_prompt: str, num_variations: int = 9) -> list[str]:
    """Génère des variantes du prompt en parallèle."""
    tasks = [(i, current_prompt, TARGET_TOKENS) for i in range(num_variations)]

    with ThreadPoolExecutor(max_workers=min(num_variations, MAX_WORKERS)) as executor:
        futures = {executor.submit(_generate_single_prompt, t): t[0] for t in tasks}
        prompts = []
        for f in as_completed(futures):
            try:
                prompts.append(f.result())
            except RuntimeError as e:
                print(f"    Erreur: {e}")

    if len(prompts) < num_variations:
        print(f"    Attention: seulement {len(prompts)}/{num_variations} prompts générés")

    random.shuffle(prompts)
    return prompts


def evaluate_prompt(prompt: str, questions: list[dict], delay_between: float = 1.0) -> dict:
    """Évalue un prompt en l'utilisant comme préfixe aux questions MMLU."""
    correct = 0
    total_tokens = 0
    per_question = []

    for i, q in enumerate(questions):
        # Construire le prompt avec le préfixe
        choices_text = "\n".join([
            f"{'ABCD'[j]}. {c}" for j, c in enumerate(q["choices"])
        ])
        full_prompt = f"""{prompt}

{q['question']}

{choices_text}"""

        # Pad à 1000 tokens
        full_prompt = pad_prompt_to_tokens(full_prompt, TARGET_TOKENS)

        # Appeler l'API
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0,
            "max_tokens": 500,
        }

        answer_labels = ["A", "B", "C", "D"]
        correct_answer = answer_labels[q["answer"]]

        for attempt in range(3):
            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                content = (message.get("content") or "").strip().upper()
                usage = data.get("usage", {})
                tokens = usage.get("total_tokens", 0)
                total_tokens += tokens

                model_answer = None
                for char in content:
                    if char in "ABCD":
                        model_answer = char
                        break

                is_correct = model_answer == correct_answer
                if is_correct:
                    correct += 1

                per_question.append({
                    "subject": q["subject"],
                    "question": q["question"][:100],
                    "expected": correct_answer,
                    "got": model_answer,
                    "correct": is_correct,
                    "tokens": tokens,
                })
                break

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    time.sleep(wait)
                    continue
                if attempt < 2:
                    time.sleep(2)
                    continue
                per_question.append({
                    "subject": q["subject"],
                    "question": q["question"][:100],
                    "expected": correct_answer,
                    "got": None,
                    "correct": False,
                    "tokens": 0,
                    "error": str(e),
                })
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                per_question.append({
                    "subject": q["subject"],
                    "question": q["question"][:100],
                    "expected": correct_answer,
                    "got": None,
                    "correct": False,
                    "tokens": 0,
                    "error": str(e),
                })
                break

        # Rate limiting entre les questions d'un même prompt
        if delay_between > 0 and i < len(questions) - 1:
            time.sleep(delay_between)

    accuracy = correct / len(questions) if questions else 0
    score = (accuracy * 10000) / total_tokens if total_tokens > 0 else 0

    return {
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "score": score,
        "correct": correct,
        "total": len(questions),
        "per_question": per_question,
    }


def evaluate_prompt_worker(args: tuple) -> dict:
    """Worker pour l'évaluation parallèle. Retourne (index, prompt, result)."""
    idx, prompt, questions = args
    result = evaluate_prompt(prompt, questions, delay_between=0.5)
    return {"index": idx, "prompt": prompt, "result": result}


def run_parallel_evaluations(prompts: list[str], questions: list[dict], max_workers: int = MAX_WORKERS) -> list[dict]:
    """Évalue plusieurs prompts en parallèle."""
    tasks = [(i, p, questions) for i, p in enumerate(prompts)]
    results: list[dict] = [None] * len(prompts)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_prompt_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res = future.result()
                results[res["index"]] = res
            except Exception as e:
                results[idx] = {"index": idx, "prompt": tasks[idx][1], "result": {
                    "score": 0, "accuracy": 0, "total_tokens": 0,
                    "correct": 0, "total": len(questions), "per_question": [],
                    "error": str(e),
                }}

    return results


def main():
    if not API_KEY:
        print("Erreur: OPENCODE_API_KEY manquant dans .env")
        sys.exit(1)

    print("=" * 60)
    print("  PROMPT OPTIMIZER")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Waves: {WAVES}")
    print(f"Target tokens: {TARGET_TOKENS}")
    print()

    # Charger le dataset une fois avec seed fixe pour reproductibilité
    print("Chargement du dataset MMLU...")
    questions = load_mmlu(NUM_QUESTIONS, seed=SEED)
    print(f"\n  Questions fixées (seed={SEED}):")
    for i, q in enumerate(questions):
        labels = "ABCD"
        print(f"    Q{i+1}: [{q['subject']}] {q['question'][:60]}... → {labels[q['answer']]}")
    print()

    # Historique complet
    all_evals = []  # Toutes les évaluations (prompt + score + détails)
    waves_data = []  # Données par wave
    best_prompt = ""  # Prompt vide au début
    best_score = 0.0

    # === WAVE 0 : Prompt vide + 9 variantes ===
    print("=" * 60)
    print("  WAVE 0: Initialisation")
    print("=" * 60)

    # Générer 9 variantes + prompt vide = 10 prompts à évaluer
    print(f"\n  Génération de 9 variantes...")
    new_prompts = generate_prompts("", 9)
    all_prompts = [""] + new_prompts  # prompt vide en premier
    print(f"  {len(all_prompts)} prompts à évaluer (en parallèle)...")

    # Évaluer tous en parallèle
    eval_results = run_parallel_evaluations(all_prompts, questions)

    wave_evals = []
    for res in eval_results:
        prompt = res["prompt"]
        result = res["result"]
        label = "(vide)" if not prompt else prompt[:50]
        print(f"  [{res['index']+1}/10] {label}... -> score={result['score']:.2f}")

        all_evals.append({
            "wave": 0,
            "index": res["index"],
            "prompt": prompt,
            "score": result["score"],
            "accuracy": result["accuracy"],
            "total_tokens": result["total_tokens"],
            "per_question": result["per_question"],
        })
        wave_evals.append({"prompt": prompt, "score": result["score"]})

        # Préférer un prompt non-vide en cas d'égalité
        if result["score"] > best_score or (result["score"] == best_score and prompt and not best_prompt):
            best_score = result["score"]
            best_prompt = prompt

    waves_data.append({
        "wave": 0,
        "best_score": best_score,
        "best_prompt": best_prompt[:200] if best_prompt else "(vide)",
        "num_evaluated": len(wave_evals),
        "all_scores": [e["score"] for e in wave_evals],
    })

    print(f"\n  Meilleur score wave 0: {best_score:.2f}")
    print()

    # === WAVES 1+ : Optimisation itérative ===
    for wave in range(1, WAVES + 1):
        print("=" * 60)
        print(f"  WAVE {wave}/{WAVES}")
        print("=" * 60)
        print(f"  Prompt actuel: {best_prompt[:60]}..." if best_prompt else "  Prompt actuel: (vide)")

        # Générer 10 variantes du meilleur prompt
        print(f"\n  Génération de 10 variantes...")
        new_prompts = generate_prompts(best_prompt, 10)
        print(f"  {len(new_prompts)} prompts à évaluer (en parallèle)...")

        # Évaluer tous en parallèle
        eval_results = run_parallel_evaluations(new_prompts, questions)

        wave_evals = []
        wave_best_score = best_score
        wave_best_prompt = best_prompt

        for res in eval_results:
            prompt = res["prompt"]
            result = res["result"]
            print(f"  [{res['index']+1}/10] {prompt[:50]}... -> score={result['score']:.2f}")

            all_evals.append({
                "wave": wave,
                "index": res["index"],
                "prompt": prompt,
                "score": result["score"],
                "accuracy": result["accuracy"],
                "total_tokens": result["total_tokens"],
                "per_question": result["per_question"],
            })
            wave_evals.append({"prompt": prompt, "score": result["score"]})

            if result["score"] > wave_best_score:
                wave_best_score = result["score"]
                wave_best_prompt = prompt

        # Mettre à jour le meilleur global (en cas d'égalité, garder le nouveau)
        if wave_best_score >= best_score:
            best_score = wave_best_score
            best_prompt = wave_best_prompt

        waves_data.append({
            "wave": wave,
            "best_score": wave_best_score,
            "best_prompt": wave_best_prompt[:200] if wave_best_prompt else "(vide)",
            "num_evaluated": len(wave_evals),
            "all_scores": [e["score"] for e in wave_evals],
        })

        print(f"\n  Meilleur score wave {wave}: {wave_best_score:.2f}")
        print()

    # === RÉSUMÉ FINAL ===
    print("=" * 60)
    print("  RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"  Score final: {best_score:.2f}")
    print(f"  Prompt final:")
    print(f"  {best_prompt[:200] if best_prompt else '(vide)'}...")
    print()
    print("  Historique des waves:")
    for w in waves_data:
        print(f"    Wave {w['wave']}: best={w['best_score']:.2f} | evaluated={w['num_evaluated']}")

    # Sauvegarder TOUT
    output_file = os.path.join(os.path.dirname(__file__), "optimizer_results.json")
    output = {
        "config": {
            "model": MODEL,
            "num_questions": NUM_QUESTIONS,
            "waves": WAVES,
            "target_tokens": TARGET_TOKENS,
            "seed": SEED,
        },
        "questions": [
            {
                "index": i,
                "subject": q["subject"],
                "question": q["question"],
                "choices": q["choices"],
                "answer": "ABCD"[q["answer"]],
            }
            for i, q in enumerate(questions)
        ],
        "best_score": best_score,
        "best_prompt": best_prompt,
        "waves_summary": waves_data,
        "all_evaluations": all_evals,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Tous les résultats sauvegardés: {output_file}")
    print(f"  ({len(all_evals)} évaluations au total)")


if __name__ == "__main__":
    main()
