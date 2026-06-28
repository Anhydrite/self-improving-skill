#!/usr/bin/env python3
"""Analyseur de résultats JSON du LLM Benchmark / Prompt Optimizer."""

import json
import sys
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summary(data: dict) -> None:
    """Affiche un résumé global."""
    cfg = data.get("config", {})
    print("=" * 60)
    print("  RÉSUMÉ GLOBAL")
    print("=" * 60)
    print(f"  Modèle      : {cfg.get('model', '?')}")
    print(f"  Questions   : {cfg.get('num_questions', '?')}")
    print(f"  Waves       : {cfg.get('waves', '?')}")
    print(f"  Seed        : {cfg.get('seed', '?')}")
    print(f"  Target tok. : {cfg.get('target_tokens', '?')}")
    print()
    print(f"  Score final : {data.get('best_score', 0):.4f}")
    best = data.get("best_prompt", "")
    print(f"  Meilleur prompt : {best[:120]}{'...' if len(best) > 120 else ''}")
    print()

    waves = data.get("waves_summary", [])
    if waves:
        print("  Progression des waves :")
        for w in waves:
            scores = w.get("all_scores", [])
            avg = sum(scores) / len(scores) if scores else 0
            print(f"    Wave {w['wave']}: best={w['best_score']:.4f}  avg={avg:.4f}  n={w.get('num_evaluated', '?')}")
    print()


def compare_waves(data: dict) -> None:
    """Compare les performances entre waves."""
    waves = data.get("waves_summary", [])
    if len(waves) < 2:
        print("Pas assez de waves pour comparer.")
        return

    print("=" * 60)
    print("  COMPARAISON DES WAVES")
    print("=" * 60)
    print(f"  {'Wave':<6} {'Best':>8} {'Avg':>8} {'Min':>8} {'Max':>8} {'StdDev':>8}")
    print("  " + "-" * 50)

    for w in waves:
        scores = w.get("all_scores", [])
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        mn = min(scores)
        mx = max(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        print(f"  {w['wave']:<6} {w['best_score']:>8.4f} {avg:>8.4f} {mn:>8.4f} {mx:>8.4f} {std:>8.4f}")

    print()

    # Évolution du meilleur score
    best_scores = [w.get("best_score", 0) for w in waves]
    if len(best_scores) >= 2:
        delta = best_scores[-1] - best_scores[0]
        pct = (delta / best_scores[0] * 100) if best_scores[0] > 0 else 0
        print(f"  Évolution : {best_scores[0]:.4f} → {best_scores[-1]:.4f} ({delta:+.4f}, {pct:+.1f}%)")
    print()


def wave_detail(data: dict, wave_num: int) -> None:
    """Affiche le détail d'une wave spécifique."""
    evals = data.get("all_evaluations", [])
    wave_evals = [e for e in evals if e.get("wave") == wave_num]

    if not wave_evals:
        print(f"Aucune évaluation trouvée pour la wave {wave_num}.")
        return

    print("=" * 60)
    print(f"  DÉTAIL WAVE {wave_num}")
    print("=" * 60)

    for i, e in enumerate(wave_evals):
        score = e.get("score", 0)
        acc = e.get("accuracy", 0)
        tokens = e.get("total_tokens", 0)
        prompt = e.get("prompt", "")
        prompt_display = prompt[:80] + "..." if len(prompt) > 80 else prompt or "(vide)"

        print(f"\n  [{i+1}] Score: {score:.4f} | Acc: {acc*100:.0f}% | Tokens: {tokens:,}")
        print(f"      Prompt: {prompt_display}")

        # Détail par question
        per_q = e.get("per_question", [])
        if per_q:
            correct = sum(1 for q in per_q if q.get("correct"))
            print(f"      {correct}/{len(per_q)} correct")
    print()


def by_subject(data: dict) -> None:
    """Analyse les performances par sujet."""
    evals = data.get("all_evaluations", [])
    if not evals:
        print("Aucune évaluation trouvée.")
        return

    # Prendre la dernière évaluation (meilleure)
    best_eval = max(evals, key=lambda e: e.get("score", 0))

    subject_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "tokens": 0})

    for q in best_eval.get("per_question", []):
        subj = q.get("subject", "unknown")
        subject_stats[subj]["total"] += 1
        if q.get("correct"):
            subject_stats[subj]["correct"] += 1
        subject_stats[subj]["tokens"] += q.get("tokens", 0)

    print("=" * 60)
    print("  PERFORMANCE PAR SUJET (meilleure évaluation)")
    print("=" * 60)
    print(f"  {'Sujet':<35} {'Acc':>6} {'Tok':>7}")
    print("  " + "-" * 50)

    for subj, stats in sorted(subject_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {subj:<35} {acc*100:>5.0f}% {stats['tokens']:>6}")

    total_correct = sum(s["correct"] for s in subject_stats.values())
    total_all = sum(s["total"] for s in subject_stats.values())
    total_tok = sum(s["tokens"] for s in subject_stats.values())
    print("  " + "-" * 50)
    print(f"  {'TOTAL':<35} {total_correct/total_all*100:>5.0f}% {total_tok:>6}")
    print()


def worst_questions(data: dict, n: int = 5) -> None:
    """Affiche les questions les plus souvent ratées."""
    evals = data.get("all_evaluations", [])
    if not evals:
        print("Aucune évaluation trouvée.")
        return

    # Compter les erreurs par question (index)
    question_errors: dict[str, dict] = {}

    for e in evals:
        for q in e.get("per_question", []):
            key = q.get("question", "")[:60]
            if key not in question_errors:
                question_errors[key] = {"errors": 0, "attempts": 0, "subject": "", "question": ""}
            question_errors[key]["attempts"] += 1
            if not q.get("correct"):
                question_errors[key]["errors"] += 1
            question_errors[key]["subject"] = q.get("subject", "")
            question_errors[key]["question"] = q.get("question", "")

    # Trier par taux d'erreur
    ranked = sorted(
        question_errors.items(),
        key=lambda x: x[1]["errors"] / x[1]["attempts"] if x[1]["attempts"] > 0 else 0,
        reverse=True,
    )

    print("=" * 60)
    print(f"  TOP {n} QUESTIONS LES PLUS DIFFICILES")
    print("=" * 60)

    for i, (key, stats) in enumerate(ranked[:n]):
        attempts = stats["attempts"]
        errors = stats["errors"]
        err_rate = errors / attempts * 100 if attempts > 0 else 0
        print(f"\n  [{i+1}] {stats['subject']}")
        print(f"      Taux d'erreur : {err_rate:.0f}% ({errors}/{attempts})")
        print(f"      {stats['question'][:100]}...")
    print()


def export_csv(data: dict, output: str) -> None:
    """Exporte les évaluations en CSV."""
    evals = data.get("all_evaluations", [])
    if not evals:
        print("Aucune évaluation à exporter.")
        return

    rows = []
    for e in evals:
        for q in e.get("per_question", []):
            rows.append({
                "wave": e.get("wave"),
                "eval_index": e.get("index"),
                "subject": q.get("subject", ""),
                "correct": q.get("correct", False),
                "expected": q.get("expected", ""),
                "got": q.get("got", ""),
                "tokens": q.get("tokens", 0),
                "question": q.get("question", ""),
                "prompt_score": e.get("score", 0),
            })

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exporté {len(rows)} lignes vers {output}")


def full_analysis(data: dict) -> None:
    """Analyse complète."""
    summary(data)
    compare_waves(data)

    # Meilleure wave
    waves = data.get("waves_summary", [])
    if waves:
        best_wave = max(waves, key=lambda w: w.get("best_score", 0))
        wave_detail(data, best_wave["wave"])

    by_subject(data)
    worst_questions(data)


def main():
    parser = argparse.ArgumentParser(description="Analyseur de résultats LLM Benchmark")
    parser.add_argument("file", nargs="?", default="optimizer_results.json",
                        help="Fichier JSON à analyser (défaut: optimizer_results.json)")
    parser.add_argument("--summary", action="store_true", help="Résumé global")
    parser.add_argument("--wave", type=int, help="Détail d'une wave spécifique")
    parser.add_argument("--compare", action="store_true", help="Comparer les waves")
    parser.add_argument("--subjects", action="store_true", help="Performance par sujet")
    parser.add_argument("--worst", action="store_true", help="Questions les plus difficiles")
    parser.add_argument("--export", choices=["csv"], help="Exporter les données")
    parser.add_argument("--full", action="store_true", help="Analyse complète")
    parser.add_argument("-o", "--output", default="results_export.csv", help="Fichier de sortie pour l'export")

    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Fichier non trouvé : {path}")
        sys.exit(1)

    data = load_results(str(path))

    # Si aucun flag, afficher le résumé
    if not any([args.summary, args.wave, args.compare, args.subjects, args.worst, args.export, args.full]):
        args.full = True

    if args.full:
        full_analysis(data)
    else:
        if args.summary:
            summary(data)
        if args.compare:
            compare_waves(data)
        if args.wave is not None:
            wave_detail(data, args.wave)
        if args.subjects:
            by_subject(data)
        if args.worst:
            worst_questions(data)
        if args.export:
            export_csv(data, args.output)


if __name__ == "__main__":
    main()
