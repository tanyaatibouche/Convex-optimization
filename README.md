# Convex Optimization : Implementation from scratch & Benchmark des optimiseurs Adam & variantes

Projet Convex - Sujet B : étude empirique de l'optimiseur Adam et de ses variantes sur Fashion-MNIST.

## Objectif

Comparer le comportement de **SGD, SGD+Momentum, RMSProp et Adam** sur des fonctions synthétiques et sur un vrai dataset de classification d'images, puis analyser la robustesse d'Adam via ses hyperparamètres et ses variantes (AdamW, AMSGrad).

## Structure du repo

```
├── projet_adam.ipynb             # Partie 1 — Implémentation from scratch
├── experiments_final.ipynb       # Partie 2 — Benchmark complet
├── data/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
└── figures/
    ├── fig1_quadratic.png        # Convergence sur f(x,y) = x² + 10y²
    ├── fig2_rosenbrock.png       # Convergence sur Rosenbrock (non-convexe)
    ├── fig3_fmnist.png           # Loss + accuracy sur Fashion-MNIST
    ├── fig4_speed.png            # Vitesse de convergence (temps réel)
    ├── fig5_sensitivity.png      # Sensibilité aux hyperparamètres d'Adam
    ├── fig6_confusion.png        # Matrice de confusion (Adam)
    ├── fig7_per_class.png        # Accuracy par classe
    ├── fig8_adamw_amsgrad.png    # Adam vs AdamW vs AMSGrad
    └── fig9_variance.png         # Stabilité sur plusieurs seeds
```

## Partie 1 — Implémentation from scratch (`projet_adam.ipynb`)

Adam est réimplémenté en NumPy pur, sans PyTorch, sur un problème de régression linéaire (prix immobilier ~ surface) pour bien comprendre chaque étape de l'algorithme.

Les 5 étapes implémentées manuellement :

```
g_t  = gradient sur le batch courant
m_t  = β₁ · m_{t-1} + (1 - β₁) · g_t          # 1er moment
v_t  = β₂ · v_{t-1} + (1 - β₂) · g_t²          # 2e moment
m̂_t  = m_t / (1 - β₁ᵗ)                          # bias correction
v̂_t  = v_t / (1 - β₂ᵗ)                          # bias correction
θ    = θ - α · m̂_t / (√v̂_t + ε)                # update
```

Le notebook contient aussi une visualisation de la **bias correction** ($1 - \beta^t$) qui montre pourquoi les premières itérations seraient trop petites sans cette correction.

## Dataset

**Fashion-MNIST** — Xiao et al. (2017)
- 60 000 images d'entraînement / 10 000 de test
- 10 classes de vêtements, images 28×28 pixels en niveaux de gris

Télécharger les 4 fichiers `.gz` depuis [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion) et les placer dans `data/`.

## Installation

```bash
pip install torch numpy matplotlib scikit-learn
```

## Lancer le notebook

```bash
jupyter notebook projet_adam.ipynb
jupyter notebook benchmarks_and_experiments.ipynb
```

Exécuter les cellules dans l'ordre. Les sections 5b, 5c et 5d nécessitent plusieurs re-entraînements — prévoir quelques minutes sur CPU.

## Résultats principaux

| Optimiseur | Accuracy finale (20 epochs) |
|---|---|
| Adam | ~89.3% |
| SGD+Momentum | ~87.1% |
| SGD | ~85.3% |
| RMSProp | ~82.7% |

Adam est le plus performant et le plus stable sur plusieurs seeds. Sa sensibilité aux hyperparamètres est faible (écart < 1% sur une plage de lr ×500), ce qui confirme les résultats de Kingma & Ba (2014).

## Références

- Kingma, D.P. & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980
- Loshchilov, I. & Hutter, F. (2017). *Decoupled Weight Decay Regularization*. arXiv:1711.05101
- Reddi, S.J. et al. (2018). *On the Convergence of Adam and Beyond*. ICLR 2018
- Xiao, H. et al. (2017). *Fashion-MNIST*. arXiv:1708.07747
