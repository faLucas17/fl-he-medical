# Apprentissage fédéré sécurisé par chiffrement homomorphe (données médicales)

Code du mémoire de Master 2 Génie Logiciel de **Fatou FALL** (Université Iba Der Thiam de Thiès) :
*Conception d'un système d'apprentissage fédéré sécurisé basé sur le chiffrement homomorphe pour la protection des données médicales.*

Plusieurs hôpitaux entraînent ensemble un modèle de diagnostic sans partager leurs données.
Chaque hôpital chiffre ses poids ; le serveur les additionne **sans jamais pouvoir les lire**
(il ne possède que la clé publique). Trois schémas sont comparés au même niveau de sécurité
(128 bits) : **CKKS**, **BFV** et **Paillier**.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux, macOS)
pip install -r requirements.txt
python tests/test_installation.py
python -m tests.test_systeme
```

Les jeux de données (Pima, UCI Heart Disease, UCI Chronic Kidney Disease) se placent dans `data/`
(voir `src/donnees.py`). Ils ne sont pas publiés dans ce dépôt.

## Organisation

| Fichier | Rôle |
|---|---|
| `src/donnees.py` | lecture, nettoyage et répartition des données entre hôpitaux |
| `src/modele.py` | réseau de neurones, entraînement local, évaluation |
| `src/chiffrement.py` | CKKS, BFV, Paillier : même interface (chiffrer, additionner, déchiffrer, clés, octets) |
| `src/federe.py` | simulation FedAvg en un seul programme |
| `src/repetitions.py` | 5 répétitions par expérience : moyenne, écart-type, mémoire vive |
| `src/reseau.py` | format des messages échangés sur le réseau (sans pickle) |
| `src/serveur.py` | serveur d'agrégation (clé publique seulement) |
| `src/hopital.py` | un hôpital ; l'hôpital 0 est le coordinateur qui crée les clés |
| `src/lancer_reseau.py` | lance serveur + hôpitaux sur un seul ordinateur |
| `src/figures.py`, `src/figures_rapport.py` | graphiques du mémoire |

## Utilisation

```
# Simulation (un seul programme)
python -m src.federe pima ckks

# Répétitions (moyenne ± écart-type)
python -m src.repetitions

# Système en réseau sur un seul ordinateur (un processus par participant)
python -m src.lancer_reseau --jeu pima --schema sans ckks bfv paillier

# Système en réseau sur plusieurs ordinateurs
python -m src.serveur --schema bfv --hopitaux 3                                  (machine serveur)
python -m src.hopital --jeu pima --schema bfv --indice 0 --serveur IP:5050       (hôpital coordinateur)
python -m src.hopital --jeu pima --schema bfv --indice 1 --serveur IP:5050       (après copie du dossier cles/)
python -m src.hopital --jeu pima --schema bfv --indice 2 --serveur IP:5050
```

Le dossier `cles/` contient la clé secrète : il n'est jamais publié (voir `.gitignore`)
et ne doit jamais être copié sur la machine du serveur.

## Résultats

Les mesures sont écrites dans `results/` (simulation), `results/repetitions/` et `results/reseau/`.
