# Protein-Legand-Spase-Autoencoder

Extension du projet [proteine-legand](https://github.com/AnesMaria/proteine-legand)
pour comparer **MLP**, **Sparse Autoencoder L1**, **Sparse Autoencoder Top-K** et
**Dense Autoencoder** avant un scoreur classique ou quantique. L'ancien dépôt
reste intact. Cette copie conserve son historique et sa baseline au commit
`305ea17f0a8509068fadd60ab633d26ccf39bc16`.

L'hypothèse étudiée est qu'une représentation sparse et compacte des patches
peut réduire leur redondance et améliorer le scoring quantique. **C'est une
hypothèse à tester, pas un gain de performance établi.**

## 1. Où intégrer le SAE

Le SAE reçoit les patches **déjà construits, sélectionnés et représentés** par
`PatchEncoder`, au point occupé par `AffinityModel.projection` :

```text
ESM-2 + pocket/graphe
  → extraction des patches et sélection FPS/fallback (inchangées)
  → représentation du patch p_i, dimension d = model.patch_dim
  → MLP OU SAE/Dense AE
  → dimension n_qubits
  → scoreur classique OU circuit PennyLane (inchangé)
  → score scalaire par patch
  → softmax masqué
  → somme(alpha_i * p_i), toujours sur les patches d'origine
  → concaténation avec ChemBERTa
  → tête MLP d'affinité
```

Branche SAE : `p_i → encoder → z_i sparse → Linear → circuit → score`.
Le decoder reconstruit `p_i` et sert à la loss auxiliaire. Si `latent_dim`
égale `n_qubits`, la projection légère est une identité. Le pooling n'est pas
remplacé par une moyenne des latents : son rôle reste strictement celui de la
baseline. Les patches de padding ne sont ni autoencodés ni inclus dans les losses.

## 2. Branches et conventions explicites

| Expérience | Projecteur | Scoreur | Contrôle |
|---|---|---|---|
| 1 | `mlp` | `classical` | baseline classique |
| 2 | `mlp` | `quantum` | baseline quantique |
| 3 | `sae`, `l1` | `classical` | compression sparse sans circuit |
| 4 | `sae`, `l1` | `quantum` | hypothèse principale |
| 5 | `dense_ae` | `quantum` | même AE, sans contrainte de sparsité |
| 6–7 optionnelles | `sae`, `topk` | `classical` / `quantum` | autre mécanisme de sparsité |

SAE et Dense AE partagent exactement les couches :

```text
encoder : d → hidden_dim → ReLU → latent_dim → ReLU
decoder : latent_dim → hidden_dim → ReLU → d
projection légère : latent_dim → n_qubits
```

Dense AE signifie **sans pénalité sparse et sans masque Top-K**. La ReLU
commune peut tout de même produire des zéros ; changer son activation dans le
contrôle aurait introduit un autre facteur de comparaison.

- L1 : `mean_sur_patches(sum_sur_latents(abs(z)))`. C'est la norme L1 moyenne
  par patch, et non une moyenne divisée par la dimension du latent.
- Top-K : après ReLU, seules les K plus fortes activations sont conservées.
  Les égalités sont départagées de façon stable. Le masque impose la sparsité,
  donc aucune L1 supplémentaire n'est appliquée. K/d_z est le ratio **nominal** ;
  le ratio mesuré peut être inférieur si certaines valeurs retenues sont nulles.
- Reconstruction : MSE moyenne sur tous les éléments des patches valides.
  La cible est détachée de l'autograd dans le mode end-to-end.
- L'encodeur de patches, le scoreur et la tête commencent avec les mêmes poids
  pour une même seed dans toutes les ablations. L'initialisation AE est isolée
  pour ne pas décaler le générateur aléatoire de la baseline.

Configuration initiale : `latent_dim=64`, `hidden_dim=128`, `topk=8`,
`lambda_sparse=1e-4` pour le pré-entraînement, `alpha=0.1` et `beta=1e-5`
pour le mode joint. `input_dim` est automatiquement `model.patch_dim` (256
dans la baseline), sans dimension codée en dur. Le beta illustratif de `1e-3`
provoquait un collapse des latents dans le petit test train ; avec notre norme
L1 **sommée par patch**, `1e-5` est un point de départ plus prudent, cohérent
avec `alpha * lambda_sparse`. Ces hyperparamètres restent à choisir sur
train/validation pour chaque protocole, sans sélectionner sur le test.

Le circuit, les encodeurs ESM/ChemBERTa, les loaders/splits, la préparation des
pockets et la sélection des patches sont repris sans modification. Les
hypothèses de reproduction PocketDTA/BALM, l'agrégation des doublons, les
fallbacks et les unités d'affinité restent celles de [BASELINE.md](BASELINE.md).
Les paramètres exacts de Fadoua n'ont toujours pas été fournis.

## 3. Fichiers créés ou adaptés

Nouveaux fichiers :

```text
src/sparse_autoencoder.py      # SparseAutoencoder, DenseAutoencoder, projection
src/sae_losses.py              # reconstruction, sparsité et objectif total
src/train_utils.py             # entraînement, diagnostics, provenance, transfert
pretrain_sae.py                # entraînement non supervisé sur patches de train
scripts/analyze_latents.py     # statistiques et histogramme JSON
scripts/run_experiments.py     # matrice d'ablations, datasets/splits/seeds
scripts/smoke_sae.py           # test d'optimisation hors réseau
tests/test_sae.py              # gradients, Top-K, padding, transferts, baseline
tests/fixtures/baseline_predictions.json # sorties du modèle d'origine
README.md / VALIDATION.md      # cette extension et ses vérifications
BASELINE.md / VALIDATION_BASELINE.md # documents originaux archivés
```

Adaptations minimales :

- `src/model.py` : projecteur sélectionnable, exposition des patches, sortie
  auxiliaire optionnelle ; le chemin MLP conserve ses opérations originales.
- `train.py` : CLI, pré-entraînement, losses, logs, paramètres, temps et exports.
- `evaluate.py` : mêmes checkpoints/splits, diagnostics et exports des latents.
- `config.yaml` : choix du projecteur, SAE et coefficients configurables.
- `aggregate.py` : mean ± std et refus de mélanger des expériences différentes.
- `scripts/run_seeds.sh` et `scripts/setup_cloud.sh` : nouveau projet indépendant.

`src/patch_encoder.py` n'a pas besoin d'être modifié : le modèle expose son
résultat avec `encode_patch_records`. `src/quantum_layer.py`, `datasets.py`,
`patch_selection.py`, les deux encodeurs et `pocket_data.py` sont identiques
au dépôt original. `src/engine.py` conserve les helpers de provenance de la
baseline ; la boucle AE est séparée dans `src/train_utils.py`.

## 4. Installation et données

Dans un nouveau clone CPU (Python 3.12 testé) :

```bash
git clone https://github.com/AnesMaria/Protein-Legand-Spase-Autoencoder.git
cd Protein-Legand-Spase-Autoencoder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1'
python -m pip install -r requirements.txt -c requirements-cpu.lock
python -m pip check
export OMP_NUM_THREADS=2
python -m pytest -q

python prepare_data.py --dataset davis
python prepare_embeddings.py --dataset davis
```

Sur une machine CUDA 12.1 avec pilote compatible, installer PyTorch depuis
`https://download.pytorch.org/whl/cu121`, puis `requirements.txt` **sans** le
verrou CPU. `device: auto` utilise CUDA si disponible, tandis que le simulateur
quantique analytique reste CPU. La voie CUDA n'a pas été vérifiée ici.

Dans l'environnement cloud courant, les dépendances existantes peuvent être
activées avec `source /workspace/.venvs/proteine-legand/bin/activate`, puis
`cd /workspace/Protein-Legand-Spase-Autoencoder`. Cette activation ne modifie
pas l'ancien checkout. Pour créer un venv propre au nouveau projet :
`bash scripts/setup_cloud.sh`, puis activer `/workspace/.venvs/protein-sae`.

KIBA et BindingDB utilisent les commandes inchangées :

```bash
python prepare_data.py --dataset kiba
python prepare_embeddings.py --dataset kiba
python prepare_data.py --dataset bindingdb
python prepare_embeddings.py --dataset bindingdb
```

Les données, graphes, caches des embeddings, poids et résultats ne sont pas
envoyés sur GitHub. Les scripts préparent des fichiers dans ce nouveau projet.
Les révisions des modèles et des sources restent épinglées dans la config et
les provenances. KIBA/BindingDB utilisent le fallback séquentiel quand aucune
structure externe n'est fournie ; ce n'est pas une représentation 3D inventée.

## 5. Mode B : entraînement end-to-end

```bash
# Baselines
python train.py --dataset davis --projector mlp --model classical --split random
python train.py --dataset davis --projector mlp --model quantum --split random

# SAE L1 ; expliciter ce mode évite de charger un pré-entraînement déjà présent
python train.py --dataset davis --projector sae --model classical --split random --no-pretrained
python train.py --dataset davis --projector sae --model quantum --split random --no-pretrained

# Contrôle Dense AE et Top-K
python train.py --dataset davis --projector dense_ae --model quantum --split random --no-pretrained
python train.py --dataset davis --projector sae --sparsity-mode topk --model quantum --split random --no-pretrained
```

`L_total = MSE_affinité + alpha*MSE_reconstruction + beta*L_sparsité`.
Alpha et beta sont `sae.reconstruction_weight` et `sae.sparsity_weight`.
La cible d'affinité est toujours standardisée sur le train seulement ; les
métriques sont reconverties dans les unités originales. La sélection du meilleur
checkpoint et l'early stopping utilisent uniquement la MSE d'affinité validation,
comme dans la baseline, et non un mélange dépendant des coefficients SAE.

Un SAE peut devenir presque constant ou nul : une forte sparsité n'est pas un
succès en soi. Surveiller reconstruction, variance, latents actifs et neurones
morts. Les losses auxiliaires peuvent aussi modifier l'encodeur de patches en
mode joint. `--freeze-patch-encoder` permet de fixer cette représentation dans
**toutes** les ablations d'une étude ; l'appliquer à une seule variante changerait
le protocole. Les choix initiaux et observations sont dans [VALIDATION.md](VALIDATION.md).

## 6. Mode A : pré-entraînement

```bash
python pretrain_sae.py --dataset davis --split random
python train.py --dataset davis --projector sae --model classical --split random
python train.py --dataset davis --projector sae --model quantum --split random
```

Si présent, le checkpoint compatible par défaut est découvert dans
`runs/pretrain/davis/sae_l1/random/seed_0/best.pt`. Sinon, la CLI affiche le mode
end-to-end. Pour un chemin personnalisé, utiliser `--sae-checkpoint` ou
`sae.pretrained_checkpoint`. `--no-pretrained` force le mode B.

`L_pretrain = MSE_reconstruction + lambda_sparse*L_sparsité`.
`lambda_sparse` est indépendant de beta du mode joint. Pour Top-K ou Dense AE,
la pénalité L1 est nulle.

La représentation des patches est déjà apprise par un module en amont. Par
défaut, ce module conserve son initialisation seedée et reste figé pendant le
pré-entraînement AE. Ce ne sont donc **pas des patches affinity-trained** par
défaut. Pour réutiliser un encodeur de patches entraîné, fournir un checkpoint
baseline issu du **même jeu de données, split et seed** :

```bash
python pretrain_sae.py --dataset davis --split random \
  --patch-checkpoint runs/davis/mlp/quantum/random/seed_0/best.pt
```

Le fichier de pré-entraînement sauvegarde l'AE **et** l'encodeur de patches,
plus les révisions et la provenance. La projection `latent → n_qubits` reste
initialisée pour l'entraînement d'affinité ; elle n'appartient pas à l'objectif
de reconstruction. Le chargement refuse une seed, un split, une table, un graphe,
une architecture AE ou une représentation de patches différente.

Seules les lignes du split d'affinité train sont exploitées. Elles sont
dédoublonnées par séquence/graphe pour le pré-entraînement. Une validation
interne par protéines est tirée **dans ce train**, sans utiliser les autres
partitions. En random split, une protéine peut naturellement figurer dans
plusieurs partitions d'affinité : le protocole n'est pas alors cold-protein.

Gel ou fine-tuning :

```bash
# Geler AE et patches ; la projection vers les qubits, le scoreur et la tête restent entraînables
python train.py --dataset davis --projector sae --model quantum --freeze-sae --freeze-patch-encoder
# Fine-tuner AE et patches (valeurs par défaut)
python train.py --dataset davis --projector sae --model quantum --no-freeze-sae --no-freeze-patch-encoder

python pretrain_sae.py --dataset davis --sparsity-mode topk
python train.py --dataset davis --projector sae --sparsity-mode topk --model quantum
python pretrain_sae.py --dataset davis --projector dense_ae
python train.py --dataset davis --projector dense_ae --model quantum
```

Pour une comparaison pré-entraînée contrôlée, la baseline peut charger le même
encodeur de patches sans charger le SAE :

```bash
python train.py --dataset davis --projector mlp --model quantum --freeze-patch-encoder \
  --patch-checkpoint runs/pretrain/davis/sae_l1/random/seed_0/best.pt
```

## 7. Résultats et latents

Les expériences sont séparées par label `mlp`, `sae_l1`, `sae_topk`, `dense_ae` :
`runs/<dataset>/<label>/<scorer>/<split>/seed_<n>/`. Un run existant n'est jamais
écrasé ; `--output` choisit un nouveau dossier.

Chaque run contient config, révisions, environnement, splits, meilleur
checkpoint, historique, prédictions, métriques et, pour les AE,
`latents_test.npz`. Ce NPZ contient `z [N_patches_valides,d_z]`, `row_id`,
`patch_index` (rang du patch sélectionné dans la paire), `attention`,
`protein_id`, `ligand_id` et le seuil d'activité. Pas de lignes de padding.
Les statistiques sont pondérées par patch ; une protéine peut donc apparaître
plusieurs fois lorsqu'elle est associée à plusieurs ligands.

```bash
python evaluate.py --checkpoint runs/davis/sae_l1/quantum/random/seed_0/best.pt \
  --save-latents runs/davis/sae_l1/quantum/random/seed_0/latents_evaluated.npz
python scripts/analyze_latents.py \
  --input runs/davis/sae_l1/quantum/random/seed_0/latents_test.npz
```

Métriques d'affinité : RMSE, MAE, Pearson, Spearman, CI. Diagnostics AE :
reconstruction MSE, pénalité sparse, fraction de zéros exacts, ratio sparse au
seuil configurable, nombre moyen de latents actifs, norme L1 moyenne, variance
par dimension puis moyenne, ratio nominal K/d_z. L'analyse ajoute les neurones
morts et un histogramme des activations (bornes/effectifs dans un JSON).

Les temps d'entraînement incluent les epochs train/validation et sauvegardes,
pas le chargement initial des modèles ni le pré-entraînement. Le temps
d'inférence couvre une passe du loader, encodage/cache compris. Préparer les
caches de tous les backbones gelés avant une comparaison de temps sérieuse.
Les paramètres entraînables incluent le decoder dans les variantes AE ; les
backbones gelés sont exclus. Les métriques indéfinies sont `null`.

## 8. Ablations et plusieurs seeds

Les cinq expériences de base, seeds 0 à 4, Davis random :

```bash
python scripts/run_experiments.py --datasets davis --splits random --seeds 0 1 2 3 4
```

Ajouter Top-K et tous les jeux/splits, après préparation des données :

```bash
python scripts/run_experiments.py --datasets davis kiba bindingdb \
  --splits random cold_protein cold_ligand cold_both --seeds 0 1 2 3 4 --include-topk
```

Le protocole `--protocol pretrained` pré-entraîne chaque AE sur le train de
chaque dataset/split/seed et fixe les patches de toutes les ablations. Leurs
poids initiaux sont identiques. Il ne réutilise pas le SAE d'un random split
dans Cold-Both. Les checkpoints produits par différents protocoles doivent
être placés dans des racines de sortie distinctes via une config dédiée.

```bash
python scripts/run_experiments.py --datasets davis --splits random --seeds 0 1 2 3 4 \
  --protocol pretrained --include-topk --dry-run
# Enlever --dry-run pour exécuter réellement les commandes.
python aggregate.py --root runs/davis/sae_l1/quantum/random
python aggregate.py --root runs/davis/mlp/quantum/random
# Variante simple end-to-end :
bash scripts/run_seeds.sh davis quantum random sae l1
bash scripts/run_seeds.sh davis quantum random sae topk
```

L'agrégation utilise l'écart-type d'échantillon `ddof=1` et indique les seeds
indéfinies. Elle refuse de mélanger projecteurs, modes sparse, protocoles,
gels, architectures, coefficients, budgets d'entraînement ou sources différentes.
Une seule seed ne définit pas de std. Le dry-run ne fabrique aucun résultat.

Les splits froids et le traitement des paires croisées restent ceux de la
baseline ; Cold-Both exclut et compte les paires entre partitions. La variation
entre seeds comprend celle des partitions et des initialisations. Les résultats
complets sur les trois datasets et cinq seeds restent à produire.

## 9. Validation courte, avant les benchmarks

```bash
python -m pytest -q
python scripts/smoke.py --model classical
python scripts/smoke.py --model quantum
python scripts/smoke_sae.py --projector sae --model quantum
python scripts/smoke_sae.py --projector sae --sparsity-mode topk --model quantum
python scripts/smoke_sae.py --projector dense_ae --model quantum

python scripts/make_davis_smoke.py
python scripts/run_experiments.py --config data/smoke/config.yaml \
  --datasets davis --splits random --seeds 0 1 --epochs 2 --include-topk
python pretrain_sae.py --config data/smoke/config.yaml --dataset davis --epochs 3
python train.py --config data/smoke/config.yaml --dataset davis --projector sae \
  --model quantum --freeze-sae --output runs/pretrained_smoke
```

Le sous-ensemble smoke contient 96 paires Davis choisies pour leur petite taille :
ces métriques sont **des vérifications de fonctionnement, pas un benchmark**.
Voir [VALIDATION.md](VALIDATION.md) pour les vérifications réellement réalisées,
les observations de collapse et les opérations non exécutées.
