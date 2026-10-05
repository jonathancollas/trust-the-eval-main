# Audit fonctionnel et scientifique de Trust the Eval / Meridian

**Date de l'audit :** 4 octobre 2026  
**Version auditée :** branche `work`, paquet `0.0.1`  
**Nature :** revue indépendante du code, des parcours locaux, des tests et de la
documentation ; comparaison documentaire avec l'état de l'art. Cet audit n'est
ni une certification, ni une réplication scientifique externe.

## 1. Résumé exécutif

### Verdict

Trust the Eval est un **prototype de recherche fonctionnel, inhabituellement
transparent et bien testé**, mais **pas encore un instrument validé pour prendre
seul une décision à fort enjeu**.

Son meilleur choix est conceptuel : il ne produit plus un mystérieux « score de
confiance » unique. Il traite une évaluation comme un instrument de mesure,
cherche plusieurs menaces à sa validité, expose les preuves et conserve les
incertitudes. Cela le place au-dessus de nombreux tableaux de bord d'eval centrés
sur une seule accuracy. Son profil de sensibilité des résultats, ses artefacts
traçables et sa distinction entre preuves réelles, structurelles et synthétiques
sont solides dans leur intention.

La faiblesse principale est tout aussi nette : **les 20 probes ne disposent pas
du même niveau de validation**. Neuf portent sur des propriétés calculables de
l'artefact, deux seulement revendiquent des labels humains réels, et onze probes
comportementales sont surtout étalonnées sur des modèles-jouets construits pour
manifester exactement le défaut recherché. Le résultat parfait de calibration
locale (20/20 probes, précision/rappel/spécificité de 1,00 sur 81 cas) démontre la
cohérence du code et des fixtures ; **il ne mesure pas la performance externe en
conditions réelles**.

### Appréciation synthétique

| Dimension | Appréciation | Motif principal |
|---|---|---|
| Utilité fonctionnelle | **Bonne pour l'exploration** | CLI, UI locale, imports, exports, observatoire et remédiations sont présents |
| Ingénierie / testabilité | **Bonne** | 319 tests passent hors réseau ; architecture modulaire et sans dépendance obligatoire |
| Ergonomie documentaire | **Moyenne** | documentation scientifique riche dans le code, mais README obsolète et deux produits imbriqués |
| Validité statistique | **Prometteuse mais partielle** | intervalles, bootstrap et sensibilité ; plusieurs hypothèses simplificatrices persistent |
| Validité externe des probes | **Faible à moyenne** | 11/20 probes au niveau « synthetic floor » ; un seul corpus humain principal |
| État de l'art | **Bon cadrage, couverture incomplète** | traite contamination, juges, prompts, puissance et provenance ; pas d'IRT/DIF ni de validation d'usage |
| Aptitude à la décision | **Assistance seulement** | les findings doivent ouvrir une enquête, pas certifier ni classer seuls |

**Recommandation :** utiliser l'outil dès maintenant comme *lint scientifique*
et générateur de dossiers de preuve ; ne pas utiliser ses sévérités comme
seuils d'acceptation contractuels avant validation externe multi-domaines.

## 2. Périmètre et méthode

L'audit a couvert :

1. l'installation source et le chargement du paquet ;
2. le recensement effectif des probes via le CLI ;
3. une exécution statique sur `examples/sample_eval_result.json` ;
4. la calibration synthétique embarquée ;
5. les 319 tests unitaires et d'intégration hors réseau ;
6. les adaptateurs JSON, Inspect et promptfoo, le runner, les statistiques, le
   modèle de validité, le stockage et les surfaces UI/observatoire ;
7. une comparaison aux publications et recommandations primaires listées en
   section 8.

N'ont pas été réalisés : appel payant à un fournisseur de modèle, test de charge,
audit de sécurité, étude utilisateur, réplication indépendante sur le corpus
MMLU-Redux complet, ni validation prospective sur une décision industrielle.

## 3. Ce qui fonctionne bien

### 3.1 Le bon objet est audité

Le produit audite la **crédibilité du résultat d'évaluation**, et non une notion
globale et vague de « bon modèle ». La batterie couvre les items, la présentation,
le comportement, le scoring, les statistiques, le temps et la provenance. Cette
décomposition correspond bien à une approche par menaces à la validité.

Le refus d'un score composite unique est scientifiquement sain : additionner un
doublon exact, une incertitude d'échantillonnage et une suspicion de
contamination donnerait un nombre facile à lire mais sans unité ni
interprétation stable. Le profil de sensibilité répond à la question utile :
« si l'on corrige le défaut, la conclusion sur les scores ou le classement
change-t-elle ? »

### 3.2 La transparence est meilleure que la moyenne

Chaque probe peut exposer sa méthode, ses seuils, ses limites, ses références et
les éléments fautifs. La calibration classe honnêtement les preuves en trois
niveaux : labels humains réels, propriété structurelle exacte, ou plancher
synthétique. La documentation reconnaît aussi l'absence d'accord
inter-annotateurs de MMLU-Redux, la circularité du désaccord avec un panel de
modèles et l'échec du pouvoir discriminant comme détecteur d'erreur de label.
Ces résultats négatifs explicites sont un vrai point fort scientifique.

La sortie conserve des preuves structurées plutôt que de simples badges. Le
stockage adressé par contenu, les hashes d'entrée, les enregistrements
vérifiables et les exports JSON/HTML/OpenTelemetry/SARIF rendent l'analyse
auditable et intégrable.

### 3.3 Le produit est réellement exécutable

L'outil enregistre bien **20 probes** : huit statiques et douze nécessitant un
modèle. Sans accès modèle, la batterie ne simule pas les résultats manquants :
elle marque explicitement les douze probes comme ignorées. Sur l'exemple fourni,
les huit probes statiques terminent sans erreur et produisent des preuves
exploitables.

Le runner isole les erreurs par probe, mètre et met en cache les appels modèle,
diffuse la progression et permet l'arrêt. L'absence de dépendance runtime
obligatoire facilite l'exécution locale et hors ligne. Les trois adaptateurs
couvrent un format générique et deux écosystèmes importants ; le connecteur
Hugging Face de l'UI et le mapping explicite des colonnes réduisent la friction.

### 3.4 Le socle de tests est substantiel

La suite complète passe avec `PYTHONPATH=src` : **319 tests en 39,22 s**. Elle
couvre les probes, la calibration, le stockage, les vues, l'ingestion, les
exports, les pipelines et une petite tranche de vraies données MMLU-Redux/HELM.
Le harnais manuel plus large annonce des invariants sur 117 sources ; c'est une
bonne stratégie de test différentiel, même si ce harnais n'est pas autonome et
n'a pas été rejoué pendant cet audit.

## 4. Ce qui ne va pas ou reste fragile

### 4.1 Problèmes fonctionnels et produit

#### Documentation principale incohérente — priorité haute

Le README décrit encore six probes, en annonce deux « done » et quatre « stub »,
puis qualifie le projet de scaffold pré-alpha. Le code en enregistre vingt,
toutes implémentées, et les tests en couvrent beaucoup plus que les « 188 »
annoncés. Cette divergence nuit directement à l'auditabilité : un utilisateur
ne sait pas si le README, le CLI ou l'UI constitue la référence.

**Action :** générer automatiquement la table des probes et le nombre de tests,
ou supprimer les nombres volatils ; publier une carte claire des deux surfaces
Trust the Eval (audit d'un run) et Meridian (observatoire de corpus).

#### Installation développeur peu robuste — priorité moyenne

Dans un checkout non installé, `python -m pytest -q` échoue dès la collecte avec
46 erreurs `ModuleNotFoundError`; la même commande réussit après installation ou
avec `PYTHONPATH=src`. Ce n'est pas un défaut du paquet installé, mais la
documentation de test présente la commande comme directement exécutable.

**Action :** ajouter une configuration pytest `pythonpath = ["src"]`, ou rendre
l'étape `pip install -e ".[dev]"` explicite dans `TESTING.md` et la CI locale.

#### Interopérabilité plus étroite que le discours — priorité moyenne

Les adaptateurs sont tolérants mais volontairement « best effort ». Ils
aplatissent certains chats, prennent une cible unique, convertissent quelques
codes de score et peuvent perdre rubriques, traces d'outils, plusieurs graders,
tokens/logprobs ou métadonnées de version. Un nom de fichier contenant
`promptfoo` déclenche l'adaptateur correspondant : la détection par nom plutôt
que par schéma est fragile.

**Action :** versionner un schéma canonique, valider les entrées avec des erreurs
actionnables, détecter le format par contenu, conserver les objets sources sans
perte et tester plusieurs versions réelles d'Inspect/promptfoo.

#### Exécution séquentielle et coût peu prédictible — priorité moyenne

La batterie est conceptuellement indépendante mais son runner est séquentiel.
Les probes à plusieurs votes et variantes peuvent multiplier les appels ; le
mètre rapporte le coût après coup, mais il manque une estimation avant lancement,
un budget dur, des limites de débit, des retries normalisés et un manifeste
complet de l'environnement fournisseur.

**Action :** ajouter un dry-run de coût/appels, un plafond, une concurrence
bornée et un replay scellé avec version exacte du modèle, paramètres, date,
région et empreinte de prompt.

### 4.2 Limites statistiques et scientifiques

#### La calibration parfaite est principalement une auto-vérification

Par défaut, les cas, les modèles défectueux, les seuils et les probes sont tous
produits dans le même dépôt. Avec seulement 38 cas positifs et 43 négatifs au
total, les résultats parfaits sont plausibles parce que les scénarios sont
conçus pour être séparables. Agréger les matrices des probes masque en outre
l'hétérogénéité des tâches et donne davantage de poids aux probes ayant plus de
fixtures.

**Interprétation correcte :** test de conformité interne et seuil minimal de
non-régression. **Interprétation incorrecte :** preuve de 100 % de précision sur
des modèles, domaines et fournisseurs inconnus.

**Action :** geler les seuils avant l'étude, constituer des jeux externes en
aveugle, faire annoter les cas par plusieurs humains, rapporter les résultats
par domaine/fournisseur et publier les intervalles sans KPI poolé triomphal.

#### Les preuves réelles sont trop concentrées

Deux probes sont rattachées à MMLU-Redux ; elles partagent donc la même famille
de tâche (QCM de connaissance), la même provenance et les limites de la même
annotation. Les onze probes comportementales restent au niveau synthétique.
Cela ne renseigne pas la généralisation aux réponses ouvertes, au code, aux
agents, au multilingue, aux outils ou aux environnements multimodaux.

**Action :** ajouter au moins trois campagnes externes : (1) juges avec
préférences humaines multi-annotateurs, (2) prompt/ordre/format sur plusieurs
familles de modèles et tâches, (3) contamination et drift avec connaissance des
snapshots d'entraînement ou canaris contrôlés.

#### Certains tests n'exploitent pas le plan expérimental

`sandbagging_paired` applique un test de deux proportions indépendantes aux
mêmes items posés sous deux cadres. La documentation reconnaît que McNemar exact
serait le test apparié approprié. Le runner ne conserve cependant pas les quatre
cellules discordantes nécessaires. De même, les intervalles Wilson et plusieurs
bootstraps supposent implicitement des items indépendants ; les benchmarks
contiennent souvent des grappes par sujet, source ou template.

**Action :** passer à McNemar exact, conserver les réponses appariées, proposer
des intervalles cluster-bootstrap/hiérarchiques et documenter explicitement la
population cible : benchmark fixe ou généralisation à une population d'items.

#### Les seuils de sévérité sont utiles mais non validés décisionnellement

Les seuils (`n<30`, écarts de 5/10/15 points, parts de 10/30/40 %, etc.) sont
lisibles et réglables, mais majoritairement heuristiques. Une sévérité devrait
dépendre du coût d'une erreur de décision, de la marge entre modèles et de
l'usage prévu, pas seulement d'un taux générique.

**Action :** séparer trois sorties : taille d'effet, incertitude et impact sur la
décision ; rendre les politiques de sévérité propres à un contexte, versionnées
et justifiées par une fonction de perte.

#### Le construct est encore fourni implicitement par le benchmark

L'outil vérifie très bien plusieurs propriétés de l'instrument, mais il ne peut
pas déduire si les items mesurent effectivement « raisonnement scientifique »,
« aide clinique sûre » ou le comportement réel attendu. Les catégories et leur
couverture ne remplacent ni une définition du construct, ni une analyse de
contenu par experts, ni une preuve convergente/discriminante, ni la validité
écologique.

**Action :** exiger une fiche d'objectif avant l'audit : décision, population,
construct, contexte, métrique, généralisation revendiquée, risques et baselines.
Produire ensuite un argument de validité reliant chaque finding aux inférences
qu'il fragilise.

#### Psychométrie incomplète

Cronbach α, difficulté et corrélation point-bisériale sont une bonne entrée en
matière, mais l'état de l'art va vers des modèles qui séparent difficulté des
items et capacité des systèmes, examinent l'information de mesure et testent
l'invariance. L'équivalent du fonctionnement différentiel des items (DIF) serait
particulièrement utile pour savoir si un template, une langue, un fournisseur ou
un groupe de modèles change la signification du score.

**Action :** ajouter, comme modules optionnels et non comme badges magiques, Rasch
/ IRT, courbes d'information, analyse factorielle et DIF ; exiger assez de
modèles/items et afficher les diagnostics d'identifiabilité.

#### Absence de validation humaine et opérationnelle

Le système juge la qualité d'un artefact automatisé, mais aucune étude ne montre
encore que ses utilisateurs comprennent correctement les findings, trouvent les
bons items, réduisent leurs erreurs de décision ou évitent l'automation bias. Or
une évaluation d'application complète combine tests du modèle, red teaming et
tests utilisateurs ; le benchmark automatisé n'est qu'une partie du dossier.

**Action :** mener des tests utilisateurs avec décisions contrefactuelles, mesurer
temps, accord, faux sentiment de sécurité et changements de décision ; ajouter un
workflow de revue/contre-signature plutôt qu'un simple export.

## 5. Positionnement par rapport à l'état de l'art

### 5.1 Là où l'outil est bien aligné

- **HELM et l'évaluation holistique :** l'approche multi-métriques, la
  transparence des prompts/sorties et la standardisation sont cohérentes avec
  HELM. Trust the Eval est complémentaire : il audite le run plutôt que de devenir
  un nouveau harness universel.
- **NIST AI 800-2 :** objectifs, choix du benchmark, mise en œuvre, analyse et
  reporting sont les bons niveaux. La provenance, l'incertitude et les limites
  explicites de l'outil vont dans ce sens.
- **NIST AI 800-3 :** la distinction entre performance sur un benchmark fixe et
  performance généralisée est exactement le prochain raffinement nécessaire. Le
  produit quantifie déjà l'incertitude, mais ne formalise pas toujours la cible
  d'estimation et le plan d'échantillonnage.
- **Qualité des labels :** MMLU-Redux justifie directement un audit des erreurs
  de gold et de l'ambiguïté. L'outil va plus loin en mesurant l'impact sur les
  scores et classements plutôt que de compter seulement les erreurs.
- **LLM-as-a-judge :** swap d'ordre, accord, biais de position et comparaison
  aux humains ciblent des problèmes documentés. C'est pertinent, mais la
  validation réelle de cette probe reste à faire.
- **Sensibilité aux prompts et contamination :** perturbations, variantes de
  format et ordre des options attaquent des menaces connues. L'outil a raison de
  présenter le signal de contamination comme une suspicion et non une preuve de
  présence dans le corpus d'entraînement.

### 5.2 Là où il est en retrait

- Il ne commence pas encore par un **protocole d'usage et une revendication de
  validité formalisés**, comme le demandent les approches modernes de science de
  la mesure.
- Il n'intègre pas les **tests utilisateurs et effets réels** d'une approche
  ARIA/TEVV ; il reste centré sur le benchmark automatisé.
- Sa psychométrie est **classique et descriptive**, sans IRT, information, DIF,
  invariance ou modèle hiérarchique.
- La validation est surtout **interne et rétrospective**, pas multi-laboratoires,
  prospective et préenregistrée.
- Les sorties ne proposent pas encore de **decision curve**, fonction de perte,
  analyse coût-bénéfice ou seuil contextualisé.
- Les modalités modernes — **agents, outils, trajectoires, multimodalité,
  environnements interactifs et performance longue durée** — sont hors du
  schéma principal centré item/question/réponse/score.

### 5.3 Position honnête sur le marché scientifique

Ce n'est pas un concurrent complet d'Inspect, promptfoo ou HELM : c'est une
**couche d'assurance qualité au-dessus des runs**. Ce positionnement est
différenciant et crédible. Aujourd'hui, sa valeur est maximale comme outil de
revue reproductible pour chercheurs et responsables eval ; elle est plus faible
comme produit de certification ou observatoire public faisant autorité.

## 6. Feuille de route recommandée

### P0 — rendre les affirmations exactes (0–1 mois)

1. Synchroniser README, statut des 20 probes, commandes d'installation et
   nombre de tests.
2. Renommer le KPI de calibration par défaut en « conformité aux scénarios
   embarqués » et afficher `38 positifs / 43 négatifs` au premier plan.
3. Ajouter à chaque rapport un bandeau : population auditée, probes exécutées,
   probes ignorées, nombre d'appels, données manquantes et limites d'inférence.
4. Corriger le test apparié de sandbagging et ajouter les intervalles appariés.

### P1 — solidifier le produit (1–3 mois)

1. Schéma canonique versionné et validation stricte des imports.
2. Estimation de coût, budget dur, replay et manifeste fournisseur complet.
3. Cluster bootstrap par sujet/source/template et distinction explicite
   benchmark accuracy / generalized accuracy.
4. Séparation nette des interfaces « audit d'un run » et « observatoire »,
   avec parcours et documentation propres.

### P2 — acquérir de la validité externe (3–9 mois)

1. Protocole préenregistré, seuils gelés et corpus caché.
2. Annotations indépendantes multiples avec accord et arbitrage.
3. Validation sur réponses ouvertes, code, multilingue et plusieurs familles de
   modèles, en publiant tous les échecs et intervalles.
4. Benchmark externe des probes `judge_swap`, `prompt_format_sensitivity`,
   `option_order_bias`, `contamination_perturb` et `model_drift` en priorité.

### P3 — passer de l'audit technique à la science de la mesure (6–12 mois)

1. Fiche obligatoire de construct et argument de validité par usage.
2. Modules IRT/Rasch, information, DIF et invariance, sous conditions de taille.
3. Tests utilisateurs et impact sur les décisions.
4. Support natif des trajectoires agentiques, outils et résultats multimodaux.
5. Réplication externe ou challenge communautaire en aveugle.

## 7. Critères de sortie du statut « prototype de recherche »

Le produit pourrait raisonnablement revendiquer une maturité d'instrument lorsque :

- au moins cinq probes comportementales auront une validation externe sur deux
  domaines et trois familles de modèles ;
- les performances seront rapportées sur un jeu caché, avec seuils gelés et
  intervalles clusterisés ;
- le schéma d'entrée garantira la conservation de la provenance et des données
  de scoring ;
- une étude montrera que les utilisateurs prennent de meilleures décisions sans
  sur-confiance ;
- chaque usage revendiqué aura un argument de validité, une fonction de perte et
  un processus de revue humaine ;
- une équipe externe aura reproduit au moins une campagne de bout en bout.

## 8. Références externes consultées

Toutes les pages ont été consultées le 4 octobre 2026. Les documents NIST
AI 800-2 et TEVV-Athlon cités sont des **drafts**, pas des normes finales.

1. NIST, [*Practices for Automated Benchmark Evaluations of Language
   Models*, AI 800-2, initial public draft](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-2.ipd.pdf),
   2026.
2. NIST, [*Expanding the AI Evaluation Toolbox with Statistical Models*,
   AI 800-3](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf), 2026.
3. NIST, [*ARIA Evaluation Planning Manual: Elements of ARIA-Style AI
   Evaluations*](https://doi.org/10.6028/NIST.AI.200-3), 2026.
4. NIST, [*TEVV-Athlon Framework for Evaluating AI Systems*, initial public
   draft](https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems),
   2026.
5. Liang et al., [*Holistic Evaluation of Language Models
   (HELM)*](https://arxiv.org/abs/2211.09110), 2022.
6. Gema et al., [*Are We Done with MMLU?*](https://arxiv.org/abs/2406.04127),
   2024.
7. Shi et al., [*Judging the Judges: A Systematic Study of Position Bias in
   LLM-as-a-Judge*](https://arxiv.org/abs/2406.07791), 2024.
8. Razavi et al., [*Benchmarking Prompt Sensitivity in Large Language
   Models*](https://arxiv.org/abs/2502.06065), 2025.
9. Xu et al., [*Benchmark Data Contamination of Large Language Models: A
   Survey*](https://arxiv.org/abs/2406.04244), 2024.
10. Zhu et al., [*CLEAN-EVAL: Clean Evaluation on Contaminated Large Language
    Models*](https://aclanthology.org/2024.findings-naacl.53/), 2024.

## 9. Conclusion

Le projet a déjà la bonne philosophie : preuve plutôt que badge, sensibilité
plutôt que score magique, limites explicites plutôt que certitude simulée. Le
code confirme qu'il ne s'agit plus du petit scaffold décrit par son README.

Mais la science d'un outil qui audite la science doit être plus exigeante que
ses tests logiciels. La prochaine étape n'est pas d'ajouter dix nouvelles
probes : c'est de faire échouer les probes existantes sur des données externes,
de mesurer où et pourquoi elles échouent, puis de relier leurs résultats à des
décisions humaines réelles. C'est ce passage qui transformera un excellent
prototype d'assurance qualité en instrument de mesure défendable.

## 10. Addendum après revue contradictoire

Une revue du commit `a608e61` a fourni des contre-exemples exécutables. Ils ont
été reproduits et ont conduit aux corrections suivantes :

- les répétitions de `provenance_repro` contournent désormais explicitement le
  cache et atteignent deux fois le fournisseur ;
- les probes de comportement utilisent le correcteur strict de réponse finale ;
  le correcteur permissif reste uniquement un objet d'audit/démonstration ;
- `judge_swap` permute réellement candidat et référence, et le κ humain est
  calculé sur les mêmes items annotés ;
- une absence de findings produit `not assessed`, et une couverture inférieure
  à la moitié des probes pertinentes produit `inconclusive`, jamais
  `supportable` ;
- Spearman est la corrélation de Pearson des rangs moyens et renvoie `None` sur
  un vecteur constant ; les égalités de classement sont représentées comme des
  ensembles et ne dépendent plus du nom du modèle ;
- la précision affichée utilise la largeur de Wilson, qui reste honnête pour un
  unique succès, et le test de cadrage apparié utilise McNemar exact ;
- la CLI détecte promptfoo par schéma, refuse les artefacts génériques vides et
  l'adaptateur Inspect prend en charge le champ pluriel `scores` sans le perdre.

Des tests de régression encodent chacun de ces contre-exemples. Ces corrections
retirent des erreurs bloquantes d'implémentation ; elles ne transforment pas les
signaux comportementaux en preuves causales. Les libellés ont donc aussi été
resserrés en « sensibilité à la reformulation », « sensibilité au cadrage
d'évaluation » et « stabilité du juge ». La recommandation générale de cet
audit demeure : exploration et revue humaine, pas certification automatique.
