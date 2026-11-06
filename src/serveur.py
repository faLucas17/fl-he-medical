"""Serveur d'agrégation (version en réseau).

Le serveur attend les hôpitaux, reçoit leurs poids chiffrés à chaque round,
les additionne SANS les déchiffrer (formule 3.2), puis renvoie la somme
chiffrée à tous les hôpitaux. Il ne reçoit que la clé publique : il ne peut
lire ni les poids des hôpitaux, ni le modèle global.

Utilisation :
    python -m src.serveur --schema ckks --hopitaux 3
    python -m src.serveur --schema paillier --hopitaux 4 --port 5050

Les mesures sont enregistrées dans results/reseau/<jeu>_<schéma>_serveur.csv.
"""
import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.federe import tous_les_schemas
from src.mesure import PicMemoire
from src.reseau import PORT_DEFAUT, empreinte, envoyer, recevoir

DOSSIER = Path(__file__).resolve().parent.parent / "results" / "reseau"


def accueillir(ecoute, nb_hopitaux):
    """Attend que tous les hôpitaux se présentent (message BONJOUR)."""
    connexions, presentations, cle_publique = {}, {}, None
    while len(connexions) < nb_hopitaux:
        sock, adresse = ecoute.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        entete, blocs, _, _ = recevoir(sock)
        assert entete["type"] == "BONJOUR", entete
        k = entete["indice"]
        connexions[k], presentations[k] = sock, entete
        if entete.get("coordinateur"):
            cle_publique = blocs[0] if blocs else b""
        print(f"  {entete['nom']} connecté depuis {adresse[0]} ({entete['n_k']} patients)", flush=True)
    if cle_publique is None:
        raise RuntimeError("Aucun hôpital coordinateur : la clé publique manque.")
    return connexions, presentations, cle_publique


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", required=True, choices=list(tous_les_schemas()))
    parser.add_argument("--hopitaux", type=int, required=True, help="nombre d'hôpitaux attendus")
    parser.add_argument("--port", type=int, default=PORT_DEFAUT)
    args = parser.parse_args()

    ecoute = socket.create_server(("0.0.0.0", args.port), reuse_port=False)
    print(f"Serveur {args.schema} en écoute sur le port {args.port}, "
          f"en attente de {args.hopitaux} hôpitaux...", flush=True)
    connexions, presentations, cle_publique = accueillir(ecoute, args.hopitaux)
    ordre = sorted(connexions)
    sockets = [connexions[k] for k in ordre]
    n_total = sum(presentations[k]["n_k"] for k in ordre)
    jeu = presentations[ordre[0]]["jeu"]
    nb_rounds = presentations[ordre[0]]["nb_rounds"]

    # Le serveur ne reçoit que la clé publique.
    schema = tous_les_schemas()[args.schema].depuis_cles(publique=cle_publique)
    signe = empreinte(cle_publique)
    for sock in sockets:
        envoyer(sock, {"type": "DEMARRER", "n_total": n_total, "empreinte": signe})
    print(f"Tous les hôpitaux sont là. n = {n_total} patients. Empreinte de la clé : {signe}", flush=True)

    historique = []
    with ThreadPoolExecutor(max_workers=len(sockets)) as pool, PicMemoire() as memoire:
        for numero in range(1, nb_rounds + 1):
            debut_round = time.perf_counter()
            # 1. Réception en parallèle des poids chiffrés de chaque hôpital
            recus = list(pool.map(recevoir, sockets))
            for entete, _, _, _ in recus:
                assert entete["type"] == "POIDS" and entete["round"] == numero, entete
            messages = [schema.depuis_octets(blocs) for _, blocs, _, _ in recus]
            t_reception = time.perf_counter() - debut_round

            # 2. Addition homomorphe, sans clé secrète
            debut = time.perf_counter()
            somme = schema.additionner(messages)
            t_agregation = time.perf_counter() - debut

            # 3. Renvoi de la somme chiffrée à tous les hôpitaux
            blocs_somme = schema.vers_octets(somme)
            envois = list(pool.map(lambda s: envoyer(s, {"type": "SOMME", "round": numero}, blocs_somme), sockets))

            historique.append({
                "round": numero,
                "octets_recus_par_hopital": sum(r[2] for r in recus) / len(recus),
                "t_transfert_moyen": sum(r[3] for r in recus) / len(recus),
                "t_transfert_max": max(r[3] for r in recus),
                "t_attente_round": t_reception,
                "t_agregation": t_agregation,
                "octets_envoyes_par_hopital": envois[0][0],
                "t_round_serveur": time.perf_counter() - debut_round,
            })
            print(f"  round {numero}/{nb_rounds} : {historique[-1]['octets_recus_par_hopital'] / 1024:.0f} Ko reçus "
                  f"par hôpital, agrégation {t_agregation:.3f} s", flush=True)

    for sock in sockets:
        envoyer(sock, {"type": "FIN"})
        sock.close()
    ecoute.close()

    DOSSIER.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(historique)
    df["ram_pic_mo"] = memoire.pic_mo
    df["ram_ajoutee_mo"] = memoire.ajout_mo
    fichier = DOSSIER / f"{jeu}_{args.schema}_serveur.csv"
    df.to_csv(fichier, index=False)
    print(f"Terminé. RAM maximale du serveur : {memoire.pic_mo:.0f} Mo. Mesures : {fichier}", flush=True)


if __name__ == "__main__":
    main()
