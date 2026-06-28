# Rapport d'analyse LLM Benchmark

## Résumé global

| Paramètre | Valeur |
|-----------|--------|
| Modèle | deepseek-v4-flash |
| Questions | 10 |
| Waves | 3 |
| Seed | 42 |
| Target tok. | 1000 |

**Score final :** 0.2630

**Meilleur prompt :** Answer the question correctly by choosing A, B, C, or D.

### Progression des waves

| Wave | Best | Avg | N |
|------|------|-----|---|
| 0 | 0.2013 | 0.1515 | 10 |
| 1 | 0.2630 | 0.1582 | 10 |
| 2 | 0.2630 | 0.1447 | 10 |
| 3 | 0.2630 | 0.1157 | 10 |

## Comparaison des waves

| Wave | Best | Avg | Min | Max | StdDev |
|------|------|-----|-----|-----|--------|
| 0 | 0.2013 | 0.1515 | 0.0653 | 0.2013 | 0.0427 |
| 1 | 0.2630 | 0.1582 | 0.0654 | 0.2630 | 0.0532 |
| 2 | 0.2630 | 0.1447 | 0.0662 | 0.1978 | 0.0482 |
| 3 | 0.2630 | 0.1157 | 0.0000 | 0.1948 | 0.0564 |

**Évolution :** 0.2013 → 0.2630 (+0.0617, +30.7%)


## Détail Wave 1

### Évaluation 1

- **Score :** 0.1315
- **Accuracy :** 20%
- **Tokens :** 15,210
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 2/10 correct

### Évaluation 2

- **Score :** 0.1980
- **Accuracy :** 30%
- **Tokens :** 15,152
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 3/10 correct

### Évaluation 3

- **Score :** 0.1990
- **Accuracy :** 30%
- **Tokens :** 15,072
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 3/10 correct

### Évaluation 4

- **Score :** 0.1305
- **Accuracy :** 20%
- **Tokens :** 15,330
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 2/10 correct

### Évaluation 5

- **Score :** 0.1308
- **Accuracy :** 20%
- **Tokens :** 15,291
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 2/10 correct

### Évaluation 6

- **Score :** 0.2630
- **Accuracy :** 40%
- **Tokens :** 15,209
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 4/10 correct

### Évaluation 7

- **Score :** 0.1318
- **Accuracy :** 20%
- **Tokens :** 15,178
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 2/10 correct

### Évaluation 8

- **Score :** 0.2005
- **Accuracy :** 30%
- **Tokens :** 14,961
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 3/10 correct

### Évaluation 9

- **Score :** 0.1312
- **Accuracy :** 20%
- **Tokens :** 15,241
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 2/10 correct

### Évaluation 10

- **Score :** 0.0654
- **Accuracy :** 10%
- **Tokens :** 15,297
- **Prompt :** `Answer the question correctly by choosing A, B, C, or D.`
- **Résultat :** 1/10 correct


## Performance par sujet (meilleure évaluation)

| Sujet | Acc | Tok |
|-------|-----|-----|
| programming | 40% | 15209 |
| **TOTAL** | **40%** | **15209** |

## Top 5 questions les plus difficiles

### 1. programming

- **Taux d'erreur :** 100% (40/40)
- **Question :** What will be the output of the following C++ code in text files?
    #include <stdio.h>    int main ...

### 2. programming

- **Taux d'erreur :** 100% (40/40)
- **Question :** Comment on the output of the following C code.
    #include <stdio.h>    int main()    {        int ...

### 3. programming

- **Taux d'erreur :** 100% (40/40)
- **Question :** class Solution:
  def __init__(self, n: int, blacklist: List[int]):
    _______________
    self.val...

### 4. programming

- **Taux d'erreur :** 100% (40/40)
- **Question :** What does the following expression means ? char ∗(∗(∗ a[N]) ( )) ( );...

### 5. programming

- **Taux d'erreur :** 100% (40/40)
- **Question :** class UnionFind:
  def __init__(self, n: int):
    self.count = n
    self.id = list(range(n))
    _...


