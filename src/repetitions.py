"""Répète chaque expérience plusieurs fois pour donner une moyenne et un écart-type.

Seule la graine de l'entraînement change d'une répétition à l'autre
(42, 43, 44, 45, 46). Le découpage des données entre hôpitaux reste le même.
Chaque répétition mesure aussi le pic de mémoire vive (RAM).

Utilisation :
    python -m src.repetitions                      # tout : 3 jeux x 4 schémas x 5 répétitions
    python -m src.repetitions --schemas sans ckks bfv
    python -m src.repetitions --jeux pima --repetitions 5
    python -m src.repetitions --resume             # recalcule seulement le tableau final

Le programme reprend là où il s'est arrêté : une répétition déjà présente dans
results/repetitions/brut.csv n'est pas relancée. On peut donc l'interrompre
(Ctrl+C) et le relancer plus tard. Paillier est long (environ 20 minutes par
répétition) : lancez-le de préférence la nuit.

Résultats :
    results/<jeu>_<schéma>.csv      courbe round par round de la répétition 1 (graine 42)
    results/repetitions/brut.csv    une ligne par répétition
    results/repetitions/resume.csv  moyenne et écart-type par jeu et par schéma
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from src.donnees import charger
from src.federe import enregistrer, simuler, tous_les_schemas
from src.mesure import PicMemoire

DOSSIER = Path(__file__).resolve().parent.parent / "results" / "repetitions"
BRUT = DOSSIER / "brut.csv"
RESUME = DOSSIER / "resume.csv"
JEUX = ["pima", "heart_uci", "ckd"]
ORDRE_SCHEMAS = ["sans", "ckks", "bfv", "paillier"]


def deja_fait():
    if not BRUT.exists():
        return set()
    df = pd.read_csv(BRUT)
    return set(zip(df["jeu"], df["schema"], df["repetition"]))


def une_repetition(donnees, nom_schema, repetition):
    """Lance une expérience complète et résume ses mesures en une ligne."""
    schema = tous_les_schemas()[nom_schema]()
    nb_hopitaux = len(donnees.y_hopitaux)
    debut = time.perf_counter()
    with PicMemoire() as memoire:
        historique, nb_poids = simuler(donnees, schema, graine=42 + repetition)
    duree = time.perf_counter() - debut
    if repetition == 0:  # graine 42 : met aussi à jour la courbe round par round (results/<jeu>_<schéma>.csv)
        enregistrer(historique, donnees.nom, schema)
    h = pd.DataFrame(historique)
    dernier = h.iloc[-1]
    return {
        "jeu": donnees.nom, "schema": nom_schema, "repetition": repetition,
        "graine": 42 + repetition, "nb_poids": nb_poids,
        "accuracy": dernier["accuracy"], "f1": dernier["f1"], "auc": dernier["auc"],
        # temps moyen par round ; le chiffrement est ramené à un hôpital
        "t_chiffrement_hopital": h["t_chiffrement"].mean() / nb_hopitaux,
        "t_agregation": h["t_agregation"].mean(),
        "t_dechiffrement": h["t_dechiffrement"].mean(),
        "t_entrainement_hopital": h["t_entrainement"].mean() / nb_hopitaux,
        "octets_par_message": h["octets_par_message"].mean(),
        "ram_pic_mo": memoire.pic_mo,
        "ram_ajoutee_mo": memoire.ajout_mo,
        "duree_totale_s": duree,
    }


def resumer():
    """Moyenne et écart-type de chaque mesure, par jeu et par schéma."""
    df = pd.read_csv(BRUT)
    mesures = ["accuracy", "f1", "auc", "t_chiffrement_hopital", "t_agregation",
               "t_dechiffrement", "octets_par_message", "ram_pic_mo", "ram_ajoutee_mo",
               "duree_totale_s"]
    resume = df.groupby(["jeu", "schema"])[mesures].agg(["mean", "std"])
    resume.columns = [f"{m}_{'moyenne' if s == 'mean' else 'ecart_type'}" for m, s in resume.columns]
    resume.insert(0, "nb_repetitions", df.groupby(["jeu", "schema"]).size())
    resume = resume.reset_index()
    resume["ordre"] = resume["schema"].map(ORDRE_SCHEMAS.index)
    resume["ordre_jeu"] = resume["jeu"].map(JEUX.index)
    resume = resume.sort_values(["ordre_jeu", "ordre"]).drop(columns=["ordre", "ordre_jeu"])
    resume.to_csv(RESUME, index=False)

    print("\n=== Moyenne ± écart-type ===")
    print(f"{'Jeu':<10} {'Schéma':<9} {'n':>2} | {'Accuracy':>15} | {'Chiffrement (s)':>19} | "
          f"{'RAM ajoutée (Mo)':>16} | {'Message (Ko)':>12}")
    for _, r in resume.iterrows():
        print(f"{r['jeu']:<10} {r['schema']:<9} {int(r['nb_repetitions']):>2} | "
              f"{r['accuracy_moyenne']:.3f} ± {r['accuracy_ecart_type']:.3f} | "
              f"{r['t_chiffrement_hopital_moyenne']:8.4f} ± {r['t_chiffrement_hopital_ecart_type']:7.4f} | "
              f"{r['ram_ajoutee_mo_moyenne']:7.1f} ± {r['ram_ajoutee_mo_ecart_type']:6.1f} | "
              f"{r['octets_par_message_moyenne'] / 1024:12.1f}")
    print("\nTableau enregistré dans", RESUME)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jeux", nargs="+", default=JEUX, choices=JEUX)
    parser.add_argument("--schemas", nargs="+", default=ORDRE_SCHEMAS, choices=ORDRE_SCHEMAS)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="recalcule seulement le tableau final")
    args = parser.parse_args()

    DOSSIER.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        faits = deja_fait()
        for nom_jeu in args.jeux:
            donnees = charger(nom_jeu)
            for nom_schema in args.schemas:
                for r in range(args.repetitions):
                    if (nom_jeu, nom_schema, r) in faits:
                        continue
                    print(f"\n>>> {nom_jeu} | {nom_schema} | répétition {r + 1}/{args.repetitions}", flush=True)
                    ligne = une_repetition(donnees, nom_schema, r)
                    pd.DataFrame([ligne]).to_csv(BRUT, mode="a", header=not BRUT.exists(), index=False)
                    print(f"    accuracy {ligne['accuracy']:.3f} | chiffrement {ligne['t_chiffrement_hopital']:.4f} s"
                          f" | RAM +{ligne['ram_ajoutee_mo']:.1f} Mo | durée {ligne['duree_totale_s']:.0f} s", flush=True)
    if BRUT.exists():
        resumer()


if __name__ == "__main__":
    main()
