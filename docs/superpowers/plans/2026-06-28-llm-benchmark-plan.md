# LLM Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Script Python qui évalue un modèle LLM via MMLU et retourne score + tokens.

**Architecture:** Script unique `llm_benchmark.py` qui charge la config depuis `.env`, télécharge MMLU depuis HuggingFace, interroge l'API OpenCode Go, et calcule le score.

**Tech Stack:** Python 3, requests, python-dotenv, datasets (HuggingFace)

---

## File Structure

```
/root/llm_langage/
├── llm_benchmark.py          # Script principal
├── .env                      # Configuration
├── requirements.txt          # Dépendances
└── docs/superpowers/
    ├── specs/                # Specs (existe déjà)
    └── plans/                # Ce fichier
```

---

### Task 1: Créer requirements.txt

**Files:**
- Create: `/root/llm_langage/requirements.txt`

- [ ] **Step 1: Créer le fichier requirements.txt**

```
requests>=2.31.0
python-dotenv>=1.0.0
datasets>=2.14.0
```

- [ ] **Step 2: Installer les dépendances**

Run: `pip install -r /root/llm_langage/requirements.txt`
Expected: Installation réussie

---

### Task 2: Créer le fichier .env

**Files:**
- Create: `/root/llm_langage/.env`

- [ ] **Step 1: Créer le fichier .env**

```
OPENCODE_API_KEY=sk-9ks2MDP8fZT7HdeZCFA3U8VWiwIMTH627BMK5eZcIHfRMKVkYwWQwqJN5xiHik3E
MODEL=mimo-v2.5
NUM_QUESTIONS=10
```

---

### Task 3: Implémenter le chargement de config

**Files:**
- Create: `/root/llm_langage/llm_benchmark.py`

- [ ] **Step 1: Créer le squelette du script avec chargement .env**

```python
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
TIMEOUT = 30
RATE_LIMIT_DELAY = 1.0  # secondes entre requêtes


def main():
    """Point d'entrée principal."""
    if not API_KEY:
        print("Erreur: OPENCODE_API_KEY manquant dans .env")
        sys.exit(1)

    print(f"=== LLM Benchmark ===")
    print(f"Model: {MODEL}")
    print(f"Questions: {NUM_QUESTIONS}")
    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Tester le chargement**

Run: `python /root/llm_langage/llm_benchmark.py`
Expected: Affiche "Model: mimo-v2.5" et "Questions: 10"

---

### Task 4: Implémenter le téléchargement du dataset MMLU

**Files:**
- Modify: `/root/llm_langage/llm_benchmark.py`

- [ ] **Step 1: Ajouter la fonction de téléchargement MMLU**

Ajouter après les imports :

```python
def load_mmlu(num_questions: int) -> list[dict]:
    """Charge et échantillonne le dataset MMLU depuis HuggingFace."""
    from datasets import load_dataset

    print("Téléchargement du dataset MMLU...")
    dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

    # Échantillonner aléatoirement
    if num_questions < len(dataset):
        indices = random.sample(range(len(dataset)), num_questions)
        dataset = dataset.select(indices)

    questions = []
    for item in dataset:
        questions.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],  # index 0-3
            "subject": item["subject"],
        })

    print(f"  {len(questions)} questions chargées depuis {len(dataset)} total")
    return questions
```

- [ ] **Step 2: Mettre à jour main() pour utiliser load_mmlu**

Remplacer le main() existant par :

```python
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
```

- [ ] **Step 3: Tester le téléchargement**

Run: `python /root/llm_langage/llm_benchmark.py`
Expected: Affiche "Téléchargement du dataset MMLU..." puis "10 questions chargées"

---

### Task 5: Implémenter l'appel API OpenCode Go

**Files:**
- Modify: `/root/llm_langage/llm_benchmark.py`

- [ ] **Step 1: Ajouter la fonction d'appel API**

Ajouter après load_mmlu :

```python
def call_opencode(question: dict) -> dict:
    """Interroge le modèle via l'API OpenCode Go. Retourne réponse + tokens."""
    # Formater la question QCM
    choices_text = "\n".join([
        f"{'ABCD'[i]}. {c}" for i, c in enumerate(question["choices"])
    ])
    prompt = f"""{question['question']}

{choices_text}

Réponds uniquement par la lettre de la bonne réponse (A, B, C ou D)."""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"].strip().upper()
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        # Extraire la lettre (premier caractère valide)
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

    except Exception as e:
        return {
            "model_answer": None,
            "total_tokens": 0,
            "raw_response": "",
            "error": str(e),
        }
```

- [ ] **Step 2: Tester l'appel API (une question)**

Ajouter temporairement dans main() après le chargement :

```python
    # Test API
    test = call_opencode(questions[0])
    print(f"Test API: {test}")
```

Run: `python /root/llm_langage/llm_benchmark.py`
Expected: Réponse du modèle avec tokens

---

### Task 6: Implémenter la boucle de benchmark

**Files:**
- Modify: `/root/llm_langage/llm_benchmark.py`

- [ ] **Step 1: Ajouter la fonction de benchmark**

Ajouter après call_opencode :

```python
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
```

- [ ] **Step 2: Mettre à jour main() pour utiliser run_benchmark**

Remplacer le main() par :

```python
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
```

- [ ] **Step 3: Tester le benchmark complet**

Run: `python /root/llm_langage/llm_benchmark.py`
Expected: 10 questions traitées, accuracy, tokens, score affichés

---

### Task 7: Ajouter gestion d'erreurs et retry

**Files:**
- Modify: `/root/llm_langage/llm_benchmark.py`

- [ ] **Step 1: Modifier call_opencode pour ajouter retry**

Remplacer call_opencode par :

```python
def call_opencode(question: dict, max_retries: int = 2) -> dict:
    """Interroge le modèle via l'API OpenCode Go. Retourne réponse + tokens."""
    choices_text = "\n".join([
        f"{'ABCD'[i]}. {c}" for i, c in enumerate(question["choices"])
    ])
    prompt = f"""{question['question']}

{choices_text}

Réponds uniquement par la lettre de la bonne réponse (A, B, C ou D)."""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"].strip().upper()
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
```

---

### Task 8: Test final et validation

**Files:**
- Modify: `/root/llm_langage/llm_benchmark.py`

- [ ] **Step 1: Lancer le benchmark avec 10 questions**

Run: `python /root/llm_langage/llm_benchmark.py`
Expected: Sortie complète avec accuracy, tokens, score

- [ ] **Step 2: Vérifier la sortie**

Vérifier que l'affichage correspond au format :
```
=== LLM Benchmark ===
Model: mimo-v2.5
Questions: 10

Téléchargement du dataset MMLU...
  10 questions chargées depuis 14042 total

Exécution du benchmark...
  Q1/10: abstract_algebra... ✓ (tokens: 230)
  ...

Accuracy: 70.0% (7/10)
Total tokens: 2,450
Score: 28.57
```
