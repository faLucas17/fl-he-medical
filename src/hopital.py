"""Un hôpital du système d'apprentissage fédéré (version en réseau).

Chaque hôpital est un programme séparé, qui peut tourner sur sa propre machine.
À chaque round, il :
1. entraîne le modèle global sur ses propres données ;
2. chiffre ses poids multipliés par son nombre de patients (formule 3.1) ;
3. envoie le message chiffré au serveur ;
4. reçoit la somme chiffrée, la déchiffre et la divise par n (formule 3.3).

L'hôpital d'indice 0 est le COORDINATEUR : il crée la paire de clés, garde la
clé secrète dans le dossier cles/ et envoie seulement la clé publique au
serveur. Les autres hôpitaux lisent la clé secrète dans ce même dossier.
Sur plusieurs ordinateurs, copiez le dossier cles/ par un canal sûr (clé USB,
par exemple), jamais par le serveur.

Utilisation (une commande par hôpital) :
    python -m src.hopital --jeu pima --schema ckks --indice 0 --serveur 127.0.0.1:5050
    python -m src.hopital --jeu pima --schema ckks --indice 1 --serveur 127.0.0.1:5050
    python -m src.hopital --jeu pima --schema ckks --indice 2 --serveur 127.0.0.1:5050

Pour l'expérience, chaque programme charge le jeu complet puis garde seulement
la part de son hôpital. Dans un vrai déploiement, chaque hôpital lirait
seulement son propre fichier.

Les mesures sont enregistrées dans results/reseau/<jeu>_<schéma>_hopital<k>.csv.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.donnees import charger
from src.evenements import Diffuseur
from src.federe import EPOQUES_LOCALES, NB_ROUNDS, tous_les_schemas
from src.mesure import PicMemoire
from src.modele import ReseauTabulaire, ecrire_poids, entrainer_localement, evaluer, lire_poids
from src.reseau import PORT_DEFAUT, connecter, empreinte, envoyer, recevoir

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "results" / "reseau"


def preparer_cles(classe, fichier_secret, fichier_public, coordinateur):
    """Le coordinateur crée les clés ; les autres hôpitaux les lisent."""
    if coordinateur:
        debut = time.perf_counter()
        schema = classe()
        secrete, publique = schema.exporter_cles()
        fichier_secret.parent.mkdir(parents=True, exist_ok=True)
        fichier_secret.write_bytes(secrete)
        fichier_public.write_bytes(publique)
        print(f"  Clés créées en {time.perf_counter() - debut:.1f} s", flush=True)
        return schema, publique
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jeu", required=True, choices=["pima", "heart_uci", "ckd"])
    parser.add_argument("--schema", required=True, choices=list(tous_les_schemas()))
    parser.add_argument("--indice", type=int, required=True, help="numéro de l'hôpital (0 = coordinateur)")
    parser.add_argument("--serveur", default=f"127.0.0.1:{PORT_DEFAUT}", help="adresse ip:port du serveur")
    parser.add_argument("--rounds", type=int, default=NB_ROUNDS)
    parser.add_argument("--cles", default=str(RACINE / "cles"), help="dossier des clés")
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--tableau", default=None, help="adresse du tableau de bord, ex. http://127.0.0.1:8080")
    args = parser.parse_args()
    diffuseur = Diffuseur(args.tableau, f"hopital{args.indice}")

    donnees = charger(args.jeu)
    X, y = donnees.X_hopitaux[args.indice], donnees.y_hopitaux[args.indice]
    nom = donnees.noms_hopitaux[args.indice]
    n_k = len(y)
    coordinateur = args.indice == 0
    classe = tous_les_schemas()[args.schema]
    fichier_secret = Path(args.cles) / f"{args.jeu}_{args.schema}.secret"
    fichier_public = Path(args.cles) / f"{args.jeu}_{args.schema}.publique"
    print(f"{nom} ({n_k} patients) | schéma {args.schema} | "
          f"{'coordinateur' if coordinateur else 'participant'}", flush=True)

    schema, publique = preparer_cles(classe, fichier_secret, fichier_public, coordinateur)

    # Présentation au serveur. Seul le coordinateur envoie la clé PUBLIQUE.
    sock = connecter(args.serveur)
    envoyer(sock, {"type": "BONJOUR", "indice": args.indice, "nom": nom, "n_k": n_k,
                   "jeu": args.jeu, "nb_rounds": args.rounds, "coordinateur": coordinateur},
            [publique] if coordinateur else [])
    entete, _, _, _ = recevoir(sock)
    assert entete["type"] == "DEMARRER", entete
    n_total = entete["n_total"]

    if not coordinateur:  # tout le monde est connecté : les clés du coordinateur existent
        schema = classe.depuis_cles(secrete=fichier_secret.read_bytes())
        publique = fichier_public.read_bytes()
    if empreinte(publique) != entete["empreinte"]:
        raise RuntimeError("La clé de cet hôpital ne correspond pas à celle du serveur : "
                           "recopiez le dossier cles/ du coordinateur.")

    # Même modèle de départ que la simulation (même graine).
    torch.manual_seed(args.graine)
    modele_global = ReseauTabulaire(X.shape[1])
    poids_global = lire_poids(modele_global)
    nb_poids = len(poids_global)
    etat = lambda e, r: diffuseur.envoyer("etat", indice=args.indice, nom=nom, etat=e, round=r)


    historique = []
    with PicMemoire() as memoire:
        for numero in range(1, args.rounds + 1):
            debut_round = time.perf_counter()
            modele_local = ReseauTabulaire(X.shape[1])
            ecrire_poids(modele_local, poids_global)
            etat("entraine", numero)
            debut = time.perf_counter()
            entrainer_localement(modele_local, X, y, epoques=EPOQUES_LOCALES,
                                 graine=args.graine + 1000 * numero + args.indice)
            t_entrainement = time.perf_counter() - debut

            etat("chiffre", numero)
            debut = time.perf_counter()
            message = schema.chiffrer(lire_poids(modele_local) * n_k)          # formule (3.1)
            t_chiffrement = time.perf_counter() - debut

            etat("envoie", numero)
            octets, t_envoi = envoyer(sock, {"type": "POIDS", "round": numero}, schema.vers_octets(message))
            etat("attend", numero)

            debut = time.perf_counter()
            entete, blocs, octets_recus, t_reception = recevoir(sock)
            t_attente = time.perf_counter() - debut
            if entete["type"] == "FIN":  # le serveur a arrêté l'entraînement (un hôpital ne répond plus)
                print("ARRÊT demandé par le serveur :", entete.get("erreur", ""), flush=True)
                diffuseur.envoyer("erreur", indice=args.indice, message=entete.get("erreur", ""))
                diffuseur.vider()
                raise SystemExit(2)
            assert entete["type"] == "SOMME" and entete["round"] == numero, entete
            etat("dechiffre", numero)

            debut = time.perf_counter()
            poids_global = schema.dechiffrer(schema.depuis_octets(blocs), nb_poids) / n_total  # (3.3)
            t_dechiffrement = time.perf_counter() - debut
            ecrire_poids(modele_global, poids_global)

            ligne = {"round": numero, "hopital": nom, "t_entrainement": t_entrainement,
                     "t_chiffrement": t_chiffrement, "octets_envoyes": octets, "t_envoi": t_envoi,
                     "t_attente_somme": t_attente, "t_reception_somme": t_reception,
                     "octets_recus": octets_recus, "t_dechiffrement": t_dechiffrement,
                     "t_round": time.perf_counter() - debut_round}
            if coordinateur:  # le coordinateur évalue le modèle global sur le jeu de test commun
                ligne.update(evaluer(modele_global, donnees.X_test, donnees.y_test))
            historique.append(ligne)
            diffuseur.envoyer("round_hopital", indice=args.indice, nom=nom, round=numero, octets=octets,
                              **{k: v for k, v in ligne.items() if k.startswith("t_")},
                              **({k: ligne[k] for k in ("accuracy", "f1", "auc")} if coordinateur else {}))
            print(f"  round {numero}/{args.rounds} : chiffrement {t_chiffrement:.3f} s | "
                  f"envoi {octets / 1024:.0f} Ko | round {ligne['t_round']:.2f} s"
                  + (f" | accuracy {ligne['accuracy']:.3f}" if coordinateur else ""), flush=True)

    entete, _, _, _ = recevoir(sock)
    assert entete["type"] == "FIN", entete
    sock.close()
    etat("termine", args.rounds)

    DOSSIER.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(historique)
    df["ram_pic_mo"] = memoire.pic_mo
    df["ram_ajoutee_mo"] = memoire.ajout_mo
    fichier = DOSSIER / f"{args.jeu}_{args.schema}_hopital{args.indice}.csv"
    df.to_csv(fichier, index=False)
    print(f"Terminé. RAM maximale : {memoire.pic_mo:.0f} Mo. Mesures : {fichier}", flush=True)
    diffuseur.envoyer("ram", indice=args.indice, ram_mo=memoire.pic_mo)
    diffuseur.vider()


if __name__ == "__main__":
    main()
