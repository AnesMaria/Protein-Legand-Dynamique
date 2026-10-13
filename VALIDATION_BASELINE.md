# Validation de cette session

Validation réalisée le 13 octobre 2026, sous Linux/Python 3.12 et PyTorch
2.5.1 CPU. Aucun GPU CUDA n'était disponible. Les versions exactes sont dans
`requirements-cpu.lock`. Les fichiers du projet ont été créés localement ;
aucun commit ni push GitHub n'a été effectué.

## Vérifications réussies

- Installation des dépendances puis réexécution de `scripts/setup_cloud.sh` :
  réussies ; `pip check` ne signale aucune dépendance incompatible.
- Suite finale `python -m pytest -q` : **14 tests réussis**, aucun échec ni skip.
  Deux avertissements attendus signalent les fallbacks sans structure.
  Tests : splits reproductibles sans chevauchement froid, CI contre calcul
  exhaustif, FPS sans doublons et invariant par rotation/translation,
  gradients classique/quantique, masque des patches, ordre du batch,
  features structurales optionnelles, indices de résidus, conversion nM,
  doublons, alignement PDB exact, cache/gel/chunks et gradients de vrais
  petits backbones Transformers, empreinte des graphes.
- Scripts synthétiques d'optimisation : classiques et quantiques réussis,
  avec gradients non nuls, baisse de MSE et restauration des poids.
  MSE classique 0.508368 → 0.017456 ; quantique 0.685221 → 0.035024.
  Ces nombres concernent des tenseurs synthétiques, pas des affinités réelles.
- Chargement des vrais poids ESM-2 et ChemBERTa : réussi ; embeddings
  résiduels de dimension 320 et ligands de dimension 384.
  Une vraie séquence de 1017 résidus a été encodée en deux chunks sans perte.
- Préparation Davis depuis le commit PocketDTA documenté : 30 056 lignes
  sources, 25 772 paires uniques après agrégation. 442 graphes C-alpha/P2Rank
  générés pour les identifiants sources, dont 419 avec correspondance exacte
  de la chaîne à la séquence. Les autres emploient la moyenne ESM globale.
- Préparation KIBA : 118 254 lignes sources, 117 657 paires uniques.
  Pas de PDB dans cette version des archives : fallback séquentiel signalé.
- Préparation BindingDB BALM : téléchargement public et réexécution depuis
  le commit HF épinglé réussis ; 24 700 lignes sources, 20 477 paires uniques.
- Génération des quatre splits sur chaque table complète : réussie.

| Table | Random train/val/test | Cold-Both train/val/test | Paires exclues Cold-Both |
|---|---|---|---|
| Davis | 20618 / 2577 / 2577 | 17080 / 222 / 222 | 8248 |
| KIBA | 94127 / 11765 / 11765 | 76368 / 1373 / 902 | 39014 |
| BindingDB | 16383 / 2047 / 2047 | 14212 / 152 / 141 | 5972 |

Ces effectifs sont ceux de la seed 0, fractions 0.8/0.1/0.1.

## Essais d'entraînement sur Davis réel

`scripts/make_davis_smoke.py` a sélectionné 12 séquences courtes et 8 ligands,
soit 96 paires réelles. Les poids préentraînés, les graphes PDB/P2Rank, le
scoring, AdamW, les logs, la validation, les checkpoints et le test ont été
exercés ensemble pendant deux epochs.

| Variante | Split | Train/val/test | Résultat technique |
|---|---|---|---|
| Classique | random | 58 / 19 / 19 | réussi |
| Quantique | random | 58 / 19 / 19 | réussi |
| Classique | cold_both | 48 / 2 / 2 | réussi |
| Quantique | cold_both | 48 / 2 / 2 | réussi |

`evaluate.py` a reproduit les métriques test des deux checkpoints random
après rechargement. La répétition des runs seed 0 a donné les mêmes pertes.
Un run classique random supplémentaire seed 1 et l'agrégation seed 0/1 avec
écart-type `ddof=1` ont aussi réussi.

Les résultats détaillés se trouvent dans `runs/davis_smoke/` sur la machine.
Ils ne sont pas inclus dans l'archive de code. **Ces essais sont des tests de
fonctionnement sur un sous-ensemble biaisé, pas des résultats scientifiques.**
Les corrélations sur deux points en Cold-Both n'ont aucune valeur probante
pour la qualité prédictive.

## Corrections d'environnement appliquées

- Cache Hugging Face dirigé vers le dossier du projet, accessible en écriture.
- PyArrow 26 nécessitait NumPy >=2 et empêchait l'import de `datasets` avec
  NumPy 1.26.4. Épinglage de PyArrow 19.0.1, puis nouvel import et préparation
  BindingDB réussis. Aucune vérification TLS/checksum n'a été désactivée.
- Révisions des deux modèles et de BALM épinglées ; versions transitives CPU
  enregistrées dans le verrou testé.

## Configuration cloud sauvegardée

Le brouillon contient `install_script`, `start_skill` et les domaines réseau
`huggingface.co`, `*.huggingface.co`, `*.hf.co`, `*.xethub.hf.co` ; les presets
de gestionnaires de paquets existants sont conservés. Aucun secret ajouté.
La sauvegarde du brouillon est confirmée ; elle ne constitue pas une publication
ni une validation de restauration dans une nouvelle tâche. Revoir et enregistrer
les paramètres, puis publier l'environnement via l'interface si souhaité.

## Non exécuté ou non vérifié

- Entraînement de benchmark sur Davis complet et seeds 0 à 4.
- Cache intégral des embeddings Davis/KIBA/BindingDB : seuls les éléments
  nécessaires aux essais courts ont été encodés. Les poids des deux modèles
  sont déjà téléchargés ; `prepare_embeddings.py` complétera les caches.
- Entraînements KIBA et BindingDB, voie CUDA, export de pickles TorchDrug natifs.
- Reproduction exacte de l'architecture/résultats de Fadoua et comparaison aux
  folds publiés ; les hypothèses et différences sont détaillées dans README.md.
- Publication, nouveau démarrage cloud, push vers le dépôt distant.

Aucun Sparse Autoencoder n'a été ajouté.
