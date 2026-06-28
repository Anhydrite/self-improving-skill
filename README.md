# LLM Benchmark & Prompt Optimizer

Outils Python pour évaluer et optimiser les performances de modèles LLM via le benchmark MMLU (Massive Multitask Language Understanding).

## Fonctionnalités

- **Benchmark MMLU** : Évalue un modèle LLM sur des questions QCM de CodeMMLU
- **Optimiseur de prompts** : Itère sur des variantes de prompts pour maximiser le score
- **Analyse de résultats** : Outil d'analyse du JSON de sortie avec métriques détaillées

## Score

Le score est calculé comme :

```
score = (accuracy × 10000) / total_tokens
```

Un score plus élevé indique une meilleure performance (précision élevée avec peu de tokens).

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Créer un fichier `.env` :

```
OPENCODE_API_KEY=sk-...
MODEL=mimo-v2.5
NUM_QUESTIONS=10
WAVES=3
SEED=42
```

## Utilisation

### Benchmark

```bash
python llm_benchmark.py
```

### Optimiseur de prompts

```bash
python prompt_optimizer.py
```

L'optimiseur fonctionne par vagues (waves) :
1. **Wave 0** : Évalue un prompt vide + 9 variantes générées
2. **Waves 1+** : Génère et évalue 10 variantes du meilleur prompt

### Analyse des résultats

```bash
python analyze_results.py                    # Résumé global
python analyze_results.py --wave 1           # Détail d'une wave
python analyze_results.py --subject physics  # Filtrer par sujet
python analyze_results.py --compare          # Comparer les waves
python analyze_results.py --worst            # Montrer les pires questions
python analyze_results.py --export csv       # Exporter en CSV
python analyze_results.py --full             # Analyse complète
```

## Structure du projet

```
llm_langage/
├── llm_benchmark.py        # Script de benchmark principal
├── prompt_optimizer.py     # Optimiseur itératif de prompts
├── analyze_results.py      # Outil d'analyse JSON
├── requirements.txt        # Dépendances Python
├── .env                    # Configuration (non versionné)
├── optimizer_results.json  # Résultats de l'optimiseur
└── codemmlu_cache.json     # Cache local du dataset
```

## Dataset

Utilise le dataset **CodeMMLU** de HuggingFace (`Fsoft-AIC/CodeMMLU`) avec les catégories :
- `programming_syntax`, `api_frameworks`, `software_principles`
- `dbms_sql`, `others`, `code_completion`
- `fill_in_the_middle`, `code_repair`, `execution_prediction`

## License

MIT
