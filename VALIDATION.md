# Validation du projet dynamique

Tests réalisés le 13 octobre 2026 sous Linux/Python 3.12, PyTorch 2.5.1 CPU,
PennyLane 0.40.0, NumPy 1.26.4, h5py 3.12.1 et MDAnalysis 2.8.0.
Les versions complètes sont dans `requirements-dynamic-cpu.lock`.

## Sources et données réellement utilisées

Les deux dépôts MISATO officiels ont été inspectés et leurs commits sont
épinglés dans `examples/misato_kd_tiny/provenance.json`. L'exemple HDF5 contient
20 complexes avec 100 frames chacun. La table d'affinité séparée a permis de
conserver **12 complexes avec Kd positifs exacts**, en excluant Ki, IC50,
censure et labels absents.

Le dépôt contient leur petit sous-ensemble transformé : 8 frames ordonnées,
64 atomes protéiques au maximum, tous les atomes lourds du ligand, indices
atomiques originaux, provenance, checksums dérivés, attribution et licences
sources. Il est directement utilisable sans télécharger les 133 GiB du dataset
complet. Les timestamps physiques des snapshots ne sont pas disponibles dans
ce HDF5 : les indices de frames sont utilisés et l'unité est explicitement
`frame_index`.

Pour 184L, le diagnostic des vraies coordonnées montre des changements :
distance lourde minimale protéine–ligand environ 3.00–3.47 Å, nombre de contacts
du crop à 4.5 Å entre 49 et 65, RMSD du crop protéique aligné jusqu'à environ
1.23 Å. La figure et le CSV sont inclus dans le dossier d'exemple. Ce ne sont
ni des coordonnées perturbées artificiellement ni des métriques d'affinité.

## Tests automatiques

**50 tests réussis, aucun échec et aucun skip.** Les 31 tests de la baseline
statique/SAE sont conservés. Les ajouts vérifient notamment :

- gradients dans le GNN, le GRU, le SAE et le circuit PennyLane pour les cinq
  variantes principales ; checkpoint roundtrip ; batches/frames/réplicas ragged ;
- sensibilité du GRU à l'ordre des frames et invariance du contrôle mean lorsque
  la référence de patches est conservée ;
- invariance spatiale du GNN par mouvement rigide et effet réel du déplacement
  du ligand sur la prédiction ;
- identité des ancres dans le temps et agrégation des réplicas par complexe ;
- refus des structures statiques, temps non croissants, masks invalides,
  mélanges d'endpoints et identités insuffisantes pour les splits froids ;
- séparation des complexes/réplicas pour les quatre méthodes de split sur
  des fixtures synthétiques **explicitement réservées aux tests** ;
- contexte ESM/ChemBERTa avec de vrais petits backbones Transformers locaux ;
- lecture native PDB+DCD, éléments atomiques explicites et correction PBC
  sur une fixture d'I/O synthétique, sans la présenter comme une simulation MD réelle.

Les avertissements sont ceux des bibliothèques de plotting/MDAnalysis et des
fixtures statiques sans structure ; aucune assertion n'a été désactivée pour
obtenir ces résultats. Les tests synthétiques complètent les essais sur les
vraies données ci-dessous et ne remplacent pas ces derniers.

## Essais end-to-end sur MISATO réel

Les sept variantes ont terminé deux epochs sur la seed 0 :

| Variante | Agrégation temporelle | Statut |
|---|---|---|
| MLP + Classical | GRU | réussi |
| MLP + Quantum | GRU | réussi |
| SAE L1 + Classical | GRU | réussi |
| SAE L1 + Quantum | GRU | réussi |
| Dense AE + Quantum | GRU | réussi |
| SAE Top-K + Classical | GRU | réussi |
| SAE Top-K + Quantum | GRU | réussi |

Les sept agrégations ont été produites ; avec une seule seed, la std reste
indéfinie (`null`). Des runs SAE+Quantum avec `mean` et `first` ont aussi terminé.
Les latents, attentions spatiales/temporelles, métriques, paramètres et timings
ont été enregistrés. Top-K conserve au plus 8 activations sur 64, conformément
au contrôle attendu.

Le pré-entraînement du SAE dynamique a terminé deux epochs en utilisant les
groupes du train d'affinité uniquement, avec validation interne disjointe et
encodeur spatial figé. Un modèle chargeant ce checkpoint, avec SAE et graphe
gelés, a terminé l'entraînement d'affinité puis son évaluation après rechargement.
L'évaluation du checkpoint SAE+Quantum joint reproduit les métriques calculées
après entraînement. L'export et l'analyse par frame des latents ont réussi.

**Limite majeure : le split de l'exemple donne 10 train / 1 validation / 1 test.**
Les corrélations et CI du test sont indéfinis. Les erreurs sur ce seul complexe
ne permettent pas de comparer les méthodes, ni de montrer un avantage du
quantique, du SAE ou du modèle temporel. Les timings des essais concurrents
sont des diagnostics de fonctionnement, pas des benchmarks de vitesse.

## Corrections d'environnement

L'installation initiale de MDAnalysis seul avait sélectionné des dépendances
graphiques récentes entraînant une mise à jour incompatible de NumPy.
Les contraintes ont été restaurées : NumPy 1.26.4, Matplotlib 3.10.1 et
ContourPy 1.3.2, puis installation de `requirements-md.txt` avec le verrou.
`pip check` ne signale plus de conflit ; les tests et essais MD ont été relancés
avec ces versions. TLS, signatures et checksums n'ont pas été désactivés.

Les caches Hugging Face et Matplotlib utilisent des chemins accessibles en
écriture. L'ancien code des deux dépôts précédents n'a pas été modifié.

## Non vérifié ou restant à exécuter

- Dataset MISATO complet et benchmarks sur cinq seeds ; cold-ligand/both de
  l'exemple nécessite d'abord des identités chimiques précises des peptides.
- Téléchargement/format natif du benchmark MDbind cité : non vérifiés ici.
  Le dépôt homonyme de prédiction de sites n'a pas été confondu avec ce benchmark.
  L'importeur générique requiert les véritables trajectoires, topologies et labels.
- PLAS-20k complet : non téléchargé ni entraîné ; métadonnées temporelles et
  séparation des réplicas nécessaires avant import, pas une concaténation
  arbitraire de conformères.
- ATLAS apo comme dataset d'affinité : non supporté sans ligand/labels réels.
- Timestamps physiques de MISATO, préparation PBC de tous les complexes et
  alignements résidu–atome : non inventés, à fournir/vérifier selon la source.
- Voie CUDA et matériel quantique : non testés ; le circuit est simulé en CPU.
- Recherche d'hyperparamètres et démonstration scientifique de l'hypothèse : à réaliser.

Le projet constitue une baseline MD modulaire vérifiée sur un petit dataset
réel. Il ne constitue pas une reproduction de résultats publiés sur les grands
benchmarks. Les références statiques sont archivées dans `STATIC_SAE.md`,
`VALIDATION_SAE.md`, `BASELINE.md` et `VALIDATION_BASELINE.md`.
