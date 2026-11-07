"""Chargement, nettoyage et découpage des jeux de données tabulaires.

Ce module prépare trois jeux de données pour l'apprentissage fédéré :
- Pima Indians Diabetes (découpé en 3 hôpitaux) ;
- UCI Heart Disease (chaque centre d'origine devient un hôpital) ;
- Chronic Kidney Disease, UCI (découpé en 3 hôpitaux).

Les images de drépanocytose sont traitées dans un module séparé.

Étapes communes :
1. on lit le fichier et on remplace les valeurs impossibles par « manquant » ;
2. on met 20 % des patients de côté comme jeu de test commun ;
3. on répartit le reste entre les hôpitaux ;
4. on complète les valeurs manquantes (médiane) et on standardise les
   variables. Ces statistiques sont calculées sur l'ensemble des données
   d'entraînement, puis appliquées au jeu de test.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DOSSIER_DONNEES = Path(__file__).resolve().parent.parent / "data"
PART_TEST = 0.20
NB_HOPITAUX = 3


@dataclass
class DonneesFederees:
    """Données prêtes pour l'apprentissage fédéré."""

    nom: str
    noms_hopitaux: list
    X_hopitaux: list          # une matrice (patients x variables) par hôpital
    y_hopitaux: list          # un vecteur d'étiquettes (0 ou 1) par hôpital
    X_test: np.ndarray
    y_test: np.ndarray
    noms_variables: list = field(default_factory=list)
    standardiseur: object = None   # pour revenir aux unités d'origine (attaque, figures)

    def resume(self) -> str:
        """Texte court qui décrit le découpage, pour vérification."""
        lignes = [f"=== {self.nom} ===",
                  f"Variables après préparation : {len(self.noms_variables)}"]
        for nom, y in zip(self.noms_hopitaux, self.y_hopitaux):
            lignes.append(f"  {nom:<16} {len(y):>4} patients | "
                          f"{100 * y.mean():5.1f} % malades")
        lignes.append(f"  {'Jeu de test':<16} {len(self.y_test):>4} patients | "
                      f"{100 * self.y_test.mean():5.1f} % malades")
        return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 1. Lecture et nettoyage de chaque jeu de données
# ---------------------------------------------------------------------------

def _lire_pima() -> pd.DataFrame:
    df = pd.read_csv(DOSSIER_DONNEES / "pima" / "diabetes.csv")
    # Une valeur de 0 est impossible pour ces mesures : c'est une donnée manquante.
    colonnes_zero_impossible = ["Glucose", "BloodPressure", "SkinThickness",
                                "Insulin", "BMI"]
    df[colonnes_zero_impossible] = df[colonnes_zero_impossible].replace(0, np.nan)
    return df.rename(columns={"Outcome": "cible"})


FICHIERS_HEART = {  # fichier officiel de l'UCI -> nom du centre médical
    "processed.cleveland.data": "Cleveland",
    "processed.hungarian.data": "Hongrie",
    "processed.switzerland.data": "Suisse",
    "processed.va.data": "Long Beach",
}
COLONNES_HEART = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def _lire_heart_uci() -> pd.DataFrame:
    """Lit les 4 fichiers officiels de l'UCI (un par centre médical)."""
    morceaux = []
    for fichier, centre in FICHIERS_HEART.items():
        morceau = pd.read_csv(DOSSIER_DONNEES / "heart_uci" / fichier, header=None,
                              names=COLONNES_HEART, na_values="?")
        morceau["hopital"] = centre
        morceaux.append(morceau)
    df = pd.concat(morceaux, ignore_index=True)
    # Une pression artérielle ou un cholestérol égal à 0 est impossible.
    df[["trestbps", "chol"]] = df[["trestbps", "chol"]].replace(0, np.nan)
    # num = 0 : pas de maladie ; num de 1 à 4 : maladie présente.
    df["cible"] = (df.pop("num") > 0).astype(int)
    # Les variables à plusieurs catégories deviennent des colonnes 0/1.
    # Une colonne « _nan » indique une valeur manquante.
    categorielles = ["cp", "restecg", "slope", "thal"]
    df = pd.get_dummies(df, columns=categorielles, dummy_na=True, dtype=float)
    return df


def _lire_ckd() -> pd.DataFrame:
    """Lit le fichier .arff de l'UCI, qui contient plusieurs erreurs de format
    (virgules en trop, tabulations, espaces). On le lit donc à la main."""
    texte = (DOSSIER_DONNEES / "ckd" / "chronic_kidney_disease_full.arff").read_text()
    entete, donnees = texte.split("@data")
    colonnes, numeriques = [], set()
    for ligne in entete.splitlines():
        if ligne.lower().startswith("@attribute"):
            nom = ligne.split()[1].strip("'")
            colonnes.append(nom)
            if "numeric" in ligne.lower():
                numeriques.add(nom)
    lignes = []
    for ligne in donnees.strip().splitlines():
        valeurs = [v.strip() for v in ligne.split(",") if v.strip() != ""]
        if len(valeurs) == len(colonnes):
            lignes.append(valeurs)
    df = pd.DataFrame(lignes, columns=colonnes).replace("?", np.nan)
    for col in numeriques:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # sg, al et su sont des niveaux ordonnés : on les garde comme nombres.
    for col in ["sg", "al", "su"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Les variables oui/non, normal/anormal, etc. deviennent 0 ou 1.
    correspondance = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1,
                      "no": 0, "yes": 1, "good": 0, "poor": 1}
    for col in ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]:
        df[col] = df[col].map(correspondance)
    df["cible"] = (df.pop("class") == "ckd").astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Découpage : jeu de test commun, puis répartition entre hôpitaux
# ---------------------------------------------------------------------------

def _decouper_aleatoire(df, graine):
    """Pima et CKD : test de 20 %, puis 3 hôpitaux de même taille,
    avec la même proportion de malades (répartition IID)."""
    entrainement, test = train_test_split(df, test_size=PART_TEST,
                                          stratify=df["cible"], random_state=graine)
    plis = StratifiedKFold(n_splits=NB_HOPITAUX, shuffle=True, random_state=graine)
    parts = [entrainement.iloc[indices]
             for _, indices in plis.split(entrainement, entrainement["cible"])]
    noms = [f"Hôpital {i + 1}" for i in range(NB_HOPITAUX)]
    return noms, parts, test


def _decouper_par_centre(df, graine):
    """Heart UCI : chaque centre d'origine devient un hôpital. On met de côté
    20 % des patients de chaque centre pour former le jeu de test commun."""
    noms, parts, tests = [], [], []
    for centre, groupe in df.groupby("hopital", sort=True):
        entrainement, test = train_test_split(groupe, test_size=PART_TEST,
                                              stratify=groupe["cible"],
                                              random_state=graine)
        noms.append(str(centre))
        parts.append(entrainement)
        tests.append(test)
    return noms, parts, pd.concat(tests)


# ---------------------------------------------------------------------------
# 3. Fonction principale
# ---------------------------------------------------------------------------

LECTEURS = {
    "pima": (_lire_pima, _decouper_aleatoire),
    "heart_uci": (_lire_heart_uci, _decouper_par_centre),
    "ckd": (_lire_ckd, _decouper_aleatoire),
}


def charger(nom: str, graine: int = 42) -> DonneesFederees:
    """Charge un jeu de données et le prépare pour l'apprentissage fédéré.

    nom : "pima", "heart_uci" ou "ckd".
    graine : nombre qui fixe le hasard, pour des résultats reproductibles.
    """
    if nom not in LECTEURS:
        raise ValueError(f"Jeu inconnu : {nom}. Choix possibles : {list(LECTEURS)}")
    lire, decouper = LECTEURS[nom]
    df = lire()
    noms, parts, test = decouper(df, graine)

    colonnes = [c for c in df.columns if c not in ("cible", "hopital")]
    # Statistiques de préparation calculées sur tout l'entraînement,
    # jamais sur le jeu de test.
    entrainement_total = pd.concat(parts)[colonnes].to_numpy(dtype=float)
    completeur = SimpleImputer(strategy="median").fit(entrainement_total)
    standardiseur = StandardScaler().fit(completeur.transform(entrainement_total))

    def preparer(partie):
        X = partie[colonnes].to_numpy(dtype=float)
        X = standardiseur.transform(completeur.transform(X))
        return X.astype(np.float32), partie["cible"].to_numpy(dtype=np.int64)

    X_hop, y_hop = zip(*(preparer(p) for p in parts))
    X_test, y_test = preparer(test)
    return DonneesFederees(nom=nom, noms_hopitaux=noms, X_hopitaux=list(X_hop),
                           y_hopitaux=list(y_hop), X_test=X_test, y_test=y_test,
                           noms_variables=colonnes, standardiseur=standardiseur)


if __name__ == "__main__":
    for nom_jeu in LECTEURS:
        try:
            print(charger(nom_jeu).resume(), "\n")
        except FileNotFoundError as erreur:
            print(f"=== {nom_jeu} === FICHIER INTROUVABLE : {erreur.filename}\n")