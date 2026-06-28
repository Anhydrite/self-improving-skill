# LLM Benchmark Script - Design Spec

## Objectif

Script Python qui évalue un modèle LLM via le benchmark MMLU public et retourne un score basé sur la consommation de tokens.

## Fonctionnalités

### Entrées
- **Modèle :** Variable configurable (défaut: `mimo-v2.5`)
- **API :** OpenCode Go (`https://opencode.ai/zen/go/v1/chat/completions`)
- **Clé API :** Fichier `.env` (`OPENCODE_API_KEY`)
- **Nombre de questions :** Configurable (défaut: 10)

### Sorties
- **Score :** `accuracy * 10000 / total_tokens` (float)
- **Tokens totaux :** Entier
- **Output :** Console simple

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  llm_benchmark.py                   │
├─────────────────────────────────────────────────────┤
│ 1. Charger .env → API_KEY, MODEL, NUM_QUESTIONS     │
│ 2. Télécharger dataset MMLU (cais/mmlu)             │
│ 3. Sélectionner N questions aléatoires              │
│ 4. Pour chaque question:                            │
│    - POST https://opencode.ai/zen/go/v1/chat/...    │
│    - Parser réponse + tokens usage                  │
│ 5. Calculer accuracy                                │
│ 6. Score = accuracy * 10000 / total_tokens          │
│ 7. Output console: score + tokens                   │
└─────────────────────────────────────────────────────┘
```

## API OpenCode Go

### Endpoint
```
POST https://opencode.ai/zen/go/v1/chat/completions
```

### Headers
```
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

### Body
```json
{
  "model": "mimo-v2.5",
  "messages": [
    {"role": "user", "content": "Question MMLU avec choix A/B/C/D"}
  ],
  "temperature": 0
}
```

### Response (à parser)
```json
{
  "choices": [{"message": {"content": "Réponse du modèle"}}],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "total_tokens": 200
  }
}
```

## Dataset MMLU

- **Source :** HuggingFace `cais/mmlu`
- **Format :** Questions QCM à 4 choix (A, B, C, D)
- **57 sujets :** abstract_algebra, anatomy, astronomy, etc.
- **Parsing :** Extraire la lettre de la réponse du modèle, comparer avec la ground truth

## Formule de Score

```
score = (accuracy * 10000) / total_tokens
```

- `accuracy` = nombre_bonnes_réponses / nombre_total_questions (0.0 - 1.0)
- `total_tokens` = somme de tous les tokens utilisés (input + output)
- Le score pénalise les modèles qui consomment beaucoup de tokens pour la même accuracy

## Fichiers

| Fichier | Rôle |
|---------|------|
| `llm_benchmark.py` | Script principal |
| `.env` | Configuration (API_KEY, MODEL, NUM_QUESTIONS) |
| `requirements.txt` | Dépendances Python |

## Configuration .env

```
OPENCODE_API_KEY=sk-9ks2MDP8fZT7HdeZCFA3U8VWiwIMTH627BMK5eZcIHfRMKVkYwWQwqJN5xiHik3E
MODEL=mimo-v2.5
NUM_QUESTIONS=10
```

## Contraintes

- Timeout par question : 30s
- Rate limiting : 1 req/sec (éviter 429)
- Gestion d'erreurs : retry 2x en cas d'échec
- Mode dry-run disponible pour tester sans appels API

## Output Console

```
=== LLM Benchmark ===
Model: mimo-v2.5
Questions: 10
Accuracy: 70.0% (7/10)
Total tokens: 2,450
Score: 28.57

Détail par question:
  Q1: ✓ (tokens: 230)
  Q2: ✗ (tokens: 215)
  ...
```
