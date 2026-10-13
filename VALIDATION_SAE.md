# Validation de l'extension SAE

Validation réalisée le 13 octobre 2026, sous Linux, Python 3.12, PyTorch 2.5.1 CPU,
Transformers 4.48.3 et PennyLane 0.40.0. Dépendances identiques à la baseline :
aucun paquet supplémentaire nécessaire. La validation de la baseline originale
est archivée dans [VALIDATION_BASELINE.md](VALIDATION_BASELINE.md).

## Ancien projet préservé

La copie a été créée dans `/workspace/Protein-Legand-Spase-Autoencoder` à partir
du commit `305ea17f0a8509068fadd60ab633d26ccf39bc16`. Le checkout et le dépôt
distant `AnesMaria/proteine-legand` sont conservés sans modification de code.
Les données/caches utiles ont été copiés, pas reliés par des symlinks. Les
chemins des graphes pointent vers leurs nouvelles copies.

Comparaison des trois CSV préparés : séquences, SMILES, identifiants et affinités
identiques à l'ancien projet. Comparaison octet par octet : `datasets.py`,
`quantum_layer.py`, `patch_encoder.py`, `patch_selection.py`, les encodeurs
ESM/ChemBERTa, `pocket_data.py` et `engine.py` restent identiques à la baseline.

## Tests automatiques

Suite finale : **31 tests réussis, zéro échec, zéro skip**. Neuf avertissements
attendus signalent le fallback sans structure sur les tenseurs de test.

Les 14 tests initiaux ont été conservés. Les nouveaux tests couvrent :

- SAE L1 et Top-K : dimensions, pertes et gradients encoder/decoder.
- Top-K : plafond de K activations, égalités déterministes, K invalide.
- MLP classique et quantique : prédictions et attention comparées aux sorties
  enregistrées depuis le véritable code du commit original.
- Initialisation commune des patches, scoreurs et têtes entre les ablations.
- Reconstruction/sparsité calculées uniquement sur les patches valides,
  gradients vers le circuit et le SAE, batch à un seul exemple.
- Projection identité quand la dimension du latent égale le nombre de qubits.
- Moments streaming, exports NPZ alignés avec les IDs/paires/patches.
- Refus des transferts entre seeds, splits ou lignes train différentes.
- Agrégation : seed exclue de la signature, coefficients contrôlés,
  refus de mélanger des losses différentes.
- AE gelé : pas de gradients sur ses paramètres, mais gradients conservés
  vers la projection et les patches quand ceux-ci restent entraînables.

Les scripts d'optimisation synthétiques L1+Quantum, Top-K+Quantum et
Dense AE+Quantum ont aussi terminé : baisse de loss, gradients et restauration
des checkpoints vérifiés. Les tenseurs synthétiques ne sont pas des données
scientifiques d'affinité.

## Essais avec les vrais encodeurs et pockets Davis

Sous-ensemble de fonctionnement : 96 paires, 12 séquences courtes, 8 ligands,
split random seedé : 58 train / 19 validation / 19 test. ESM-2, ChemBERTa et
graphes C-alpha/P2Rank réels ont été utilisés, comme dans la baseline.

Le runner end-to-end a terminé les sept variantes sur seeds **0 et 1**, deux
epochs par run : **14 runs**, plus les sept agrégations mean ± sample std.
Ces premiers essais employaient `beta=1e-3` pour L1. Les paramètres entraînables
mesurés dans ce protocole, backbones gelés :

| Variante | Paramètres entraînables |
|---|---:|
| MLP + Classical | 282738 |
| MLP + Quantum | 282742 |
| SAE L1 + Classical | 348786 |
| SAE L1 + Quantum | 348790 |
| Dense AE + Quantum | 348790 |
| SAE Top-K + Classical | 348786 |
| SAE Top-K + Quantum | 348790 |

Les AE incluent les paramètres du decoder dans ce compte ; les nombres ne
constituent pas une égalisation de budget entre MLP et AE. Dense AE et SAE
ont la même architecture/initialisation, ce qui contrôle l'effet de la sparsité.

Pré-entraînement : L1 (3 epochs), Top-K et Dense AE (2 epochs) terminés sur les
entrées du train uniquement. Une validation interne disjointe par séquence
protéique a été vérifiée. Les trois pré-entraînements conservent exactement
les mêmes poids de l'encodeur de patches en amont.

Le runner `--protocol pretrained --include-topk` a aussi terminé ses **sept
variantes seed 0**, avec des patches figés pour toutes les branches et
`beta=1e-5`, dans une racine distincte. Les AE étaient fine-tunés.
Un essai séparé avec SAE **et** patches gelés a vérifié le gel, la sauvegarde et
la restauration des métriques. Les transferts pré-entraînés Top-K et Dense AE
ont été vérifiés aussi individuellement.

`evaluate.py` a reproduit les métriques du checkpoint SAE+Quantum joint après
rechargement, avec export des latents. `analyze_latents.py` a produit les
statistiques par dimension, neurones morts et histogrammes JSON pour L1 et
Top-K. Les latents Top-K ont au plus 8 valeurs actives sur 64 ; dans ces essais
le ratio observé était 0.125, conforme au ratio nominal.

Un transfert volontaire d'un checkpoint random vers Cold-Both a été **refusé
comme prévu**. Les splits et les lignes train doivent correspondre exactement.

## Observation de collapse et choix initial du coefficient

La L1 est une norme **sommée sur les latents**, puis moyennée sur les patches.
Avec `alpha=0.1`, `beta=1e-3`, le deuxième epoch du train de l'essai L1+Quantum
seed 0 donnait seulement **0.84 latent actif sur 64** en moyenne et une variance
latente très faible. Une simple fraction élevée de zéros aurait donc été une
conclusion trompeuse.

Un essai train avec `beta=1e-5` donnait environ **26.95 latents actifs sur 64** au
deuxième epoch, avec variance conservée. Les branches classique et quantique
ont toutes deux été vérifiées avec ce coefficient. La config initiale publiée
emploie donc `beta=1e-5`, cohérent avec `alpha*lambda_sparse` pour `lambda=1e-4`.
Ce choix repose sur les diagnostics du train, pas sur un gain de RMSE du test.
Il ne remplace pas une recherche d'hyperparamètres et une surveillance des
neurones morts sur les véritables expériences.

## Résultats et limites

Les logs, timings, checkpoints, NPZ et JSON détaillés sont conservés dans les
dossiers ignorés `runs/` de la machine. Les scripts enregistrent les temps de
train/inférence, mais ces petites mesures CPU et les exécutions concurrentes
de validation ne conviennent pas à une comparaison de vitesse scientifique.
Les timings des benchmarks devront être mesurés sans concurrence, avec les
caches déjà préparés et en comptant séparément le pré-entraînement.

**Les essais courts ne montrent pas que le SAE améliore l'affinité.** Ils
valident l'intégration, ses losses, les contrôles et les exports. Le sous-ensemble
est biaisé par la sélection des petites séquences/molécules et ne représente
pas le benchmark Davis complet.

Non exécuté dans cette extension : benchmarks complets Davis/KIBA/BindingDB,
seeds 0 à 4 au complet, quatre splits sur tous les datasets, voie CUDA et
reproduction exacte des résultats de Fadoua. Les données et les commandes pour
ces protocoles sont disponibles. La matrice complète pré-entraînée avec Top-K
a été générée en **dry-run** (684 commandes), sans prétendre les avoir exécutées.
