"""Lance le système complet sur cet ordinateur : 1 serveur + 1 programme par hôpital.

Chaque participant est un processus séparé qui communique par le réseau
(adresse 127.0.0.1). C'est l'étape A du déploiement. Pour l'étape B
(plusieurs ordinateurs), lancez src.serveur et src.hopital à la main.

Utilisation :
    python -m src.lancer_reseau --jeu pima --schema ckks
    python -m src.lancer_reseau --jeu pima --schema sans ckks bfv
    python -m src.lancer_reseau --tous                 # 3 jeux x 4 schémas (Paillier : long)
    python -m src.lancer_reseau --resume               # recalcule seulement le tableau final

À la fin, le programme compare l'accuracy obtenue en réseau à celle de la
simulation (results/<jeu>_<schéma>.csv) et écrit results/reseau/resume.csv.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.donnees import charger
from src.reseau import PORT_DEFAUT

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "results" / "reseau"
NOMS_FICHIERS = {"sans": "sans_chiffrement", "ckks": "ckks", "bfv": "bfv", "paillier": "paillier"}
JEUX = ["pima", "heart_uci", "ckd"]
SCHEMAS = ["sans", "ckks", "bfv", "paillier"]


def lancer(jeu, schema, port, rounds, tableau=None, pause=0.0):
    nb_hopitaux = len(charger(jeu).y_hopitaux)
    journaux = DOSSIER / "journaux"
    journaux.mkdir(parents=True, exist_ok=True)
    python = [sys.executable, "-m"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")  # accents corrects dans les journaux (Windows)
    print(f"\n>>> {jeu} | {schema} | 1 serveur + {nb_hopitaux} hôpitaux", flush=True)
    debut = time.perf_counter()
    with open(journaux / f"{jeu}_{schema}_serveur.txt", "w", encoding="utf-8") as j:
        serveur = subprocess.Popen(python + ["src.serveur", "--schema", schema, "--hopitaux",
                                             str(nb_hopitaux), "--port", str(port), "--pause", str(pause)]
                                   + (["--tableau", tableau] if tableau else []),
                                   cwd=RACINE, env=env, stdout=j, stderr=subprocess.STDOUT)
    time.sleep(2)
    hopitaux = []
    for k in range(nb_hopitaux):
        j = open(journaux / f"{jeu}_{schema}_hopital{k}.txt", "w", encoding="utf-8")
        hopitaux.append((subprocess.Popen(python + ["src.hopital", "--jeu", jeu, "--schema", schema,
                                                    "--indice", str(k), "--serveur", f"127.0.0.1:{port}",
                                                    "--rounds", str(rounds)]
                                          + (["--tableau", tableau] if tableau else []),
                                          cwd=RACINE, env=env, stdout=j, stderr=subprocess.STDOUT), j))
    codes = [p.wait() for p, _ in hopitaux] + [serveur.wait()]
    for _, j in hopitaux:
        j.close()
    if any(codes):
        print(f"    ERREUR : voir les journaux dans {journaux}", flush=True)
        return False
    print(f"    terminé en {time.perf_counter() - debut:.0f} s", flush=True)
    return True


def resumer():
    lignes = []
    for jeu in JEUX:
        for schema in SCHEMAS:
            f_serveur = DOSSIER / f"{jeu}_{schema}_serveur.csv"
            f_coord = DOSSIER / f"{jeu}_{schema}_hopital0.csv"
            if not (f_serveur.exists() and f_coord.exists()):
                continue
            srv, coord = pd.read_csv(f_serveur), pd.read_csv(f_coord)
            hop = pd.concat(pd.read_csv(f) for f in sorted(DOSSIER.glob(f"{jeu}_{schema}_hopital*.csv")))
            f_simu = RACINE / "results" / f"{jeu}_{NOMS_FICHIERS[schema]}.csv"
            acc_simu = pd.read_csv(f_simu)["accuracy"].iloc[len(coord) - 1] if f_simu.exists() else float("nan")
            lignes.append({
                "jeu": jeu, "schema": schema, "nb_hopitaux": hop["hopital"].nunique(),
                "accuracy_reseau": coord["accuracy"].iloc[-1], "auc_reseau": coord["auc"].iloc[-1],
                "accuracy_simulation": acc_simu,
                "message_ko": srv["octets_recus_par_hopital"].mean() / 1024,
                "t_chiffrement_hopital": hop["t_chiffrement"].mean(),
                "t_transfert_mesure": srv["t_transfert_moyen"].mean(),
                "t_agregation": srv["t_agregation"].mean(),
                "t_dechiffrement": hop["t_dechiffrement"].mean(),
                "t_round": coord["t_round"].mean(),
                "ram_hopital_mo": hop["ram_pic_mo"].max(),
                "ram_serveur_mo": srv["ram_pic_mo"].max(),
                "ram_ajoutee_hopital_mo": hop.get("ram_ajoutee_mo", pd.Series(dtype=float)).max(),
                "ram_ajoutee_serveur_mo": srv.get("ram_ajoutee_mo", pd.Series(dtype=float)).max(),
            })
    if not lignes:
        print("Aucun résultat en réseau pour l'instant.")
        return
    df = pd.DataFrame(lignes)
    df.to_csv(DOSSIER / "resume.csv", index=False)
    print("\n=== Système en réseau : résumé ===")
    print(f"{'Jeu':<10} {'Schéma':<9} | {'Acc. réseau':>11} | {'Acc. simu':>9} | {'Message':>10} | "
          f"{'Chiffr. (s)':>11} | {'Transfert (s)':>13} | {'Round (s)':>9} | {'RAM hôp.':>8}")
    for _, r in df.iterrows():
        print(f"{r['jeu']:<10} {r['schema']:<9} | {r['accuracy_reseau']:>11.3f} | {r['accuracy_simulation']:>9.3f} | "
              f"{r['message_ko']:>7.0f} Ko | {r['t_chiffrement_hopital']:>11.4f} | {r['t_transfert_mesure']:>13.4f} | "
              f"{r['t_round']:>9.2f} | {r['ram_hopital_mo']:>5.0f} Mo")
    print("\nTableau enregistré dans", DOSSIER / "resume.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jeu", nargs="+", default=["pima"], choices=JEUX)
    parser.add_argument("--schema", nargs="+", default=["sans", "ckks", "bfv"], choices=SCHEMAS)
    parser.add_argument("--tous", action="store_true", help="les 3 jeux et les 4 schémas")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--port", type=int, default=PORT_DEFAUT)
    parser.add_argument("--resume", action="store_true", help="recalcule seulement le tableau final")
    parser.add_argument("--tableau", default=None, help="adresse du tableau de bord, ex. http://127.0.0.1:8080")
    parser.add_argument("--pause", type=float, default=0.0, help="pause (s) entre deux rounds, pour une démonstration")
    args = parser.parse_args()
    jeux, schemas = (JEUX, SCHEMAS) if args.tous else (args.jeu, args.schema)
    if not args.resume:
        for jeu in jeux:
            for schema in schemas:
                lancer(jeu, schema, args.port, args.rounds, args.tableau, args.pause)
    resumer()


if __name__ == "__main__":
    main()
