# Petit échantillon MISATO réel

Ces 12 fichiers contiennent des **trajectoires MD réelles**, pas des conformères
statiques ni du bruit ajouté. Ils proviennent des exemples publics des projets
MISATO et MISATO-affinity, aux commits et URLs indiqués dans `provenance.json`.
Les licences LGPL-2.1 des deux dépôts sources sont jointes dans ce dossier.

Transformations réalisées le 13 octobre 2026 : conservation uniquement des Kd
positifs exacts (`=`), conversion nM → pKd, sélection uniforme de **8 snapshots
ordonnés parmi 100**, retrait des hydrogènes, crop fixe d'au plus 64 atomes de
protéine proches du ligand sur les frames sélectionnées, conservation de tous
les atomes lourds du ligand, conversion des coordonnées en float32.
`prepare_md.py` fournit le code source de ces transformations. Les sources
complètes peuvent être récupérées avec `scripts/fetch_misato_example.py`.

`atom_ids` garde les indices atomiques originaux. `reference_positions` garde
la première frame de référence afin de fixer les ancres de patches. `time`
contient **les indices des frames sources**, pas des nanosecondes inventées.
Les coordonnées sont celles fournies par MISATO, en Å ; aucun nouveau
traitement de périodicité n'a été appliqué. Les propriétés simulées MMGBSA,
RMSD et autres ne remplacent pas les cibles expérimentales et ne sont pas utilisées
comme labels d'affinité.

Les identifiants de peptide tels que `4-mer` ne permettent pas d'établir une
identité chimique précise. Les splits cold-ligand/both demandent une table
supplémentaire de SMILES ou de groupes chimiques vérifiés. Le split random
groupe conservativement les alias protéine/ligand fournis.

Ce petit échantillon est destiné à **vérifier le fonctionnement**, pas à mesurer
la performance scientifique : avec la seed/fractions par défaut, il ne reste
qu'un complexe dans chacune des partitions validation et test.

Sources et attribution :

- Till Siebenmorgen et collaborateurs, MISATO, Nature Computational Science
  (2024), DOI `10.1038/s43588-024-00627-2`.
- [MISATO dataset](https://github.com/t7morgen/misato-dataset).
- [MISATO affinity prediction](https://github.com/kierandidi/misato-affinity).

Les fichiers sont fournis avec les notices/licences des sources ; les protocoles
et partitions officiels de leurs publications ne sont pas reproduits par ce smoke test.
