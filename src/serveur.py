"""Serveur d'agrégation (version en réseau).

Le serveur attend les hôpitaux, reçoit leurs poids chiffrés à chaque round,
les additionne SANS les déchiffrer (formule 3.2), puis renvoie la somme
chiffrée à tous les hôpitaux. Il ne reçoit que la clé publique : il ne peut
lire ni les poids des hôpitaux, ni le modèle global.

Utilisation :
    python -m src.serveur --schema ckks --hopitaux 3
    python -m src.serveur --schema paillier --hopitaux 4 --port 5050
    python -m src.serveur --schema bfv --hopitaux 3 --tableau http://127.0.0.1:8080 --pause 1.5

Options utiles :
    --tableau URL   envoie les événements au tableau de bord (src/tableau_de_bord.py)
    --pause S       attend S secondes avant de renvoyer la somme (démonstration au ralenti)
    --delai S       si un hôpital ne répond plus pendant S secondes, le serveur
                    prévient les autres hôpitaux et s'arrête proprement

Les mesures sont enregistrées dans results/reseau/<jeu>_<schéma>_serveur.csv.
"""
import argparse
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.evenements import Diffuseur, apercu_octets
from src.federe import tous_les_schemas
from src.mesure import PicMemoire
from src.reseau import PORT_DEFAUT, empreinte, envoyer, recevoir

DOSSIER = Path(__file__).resolve().parent.parent / "results" / "reseau"


def accueillir(ecoute, nb_hopitaux, diffuseur):
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
        diffuseur.envoyer("hopital_connecte", indice=k, nom=entete["nom"], n_k=entete["n_k"],
                          adresse=adresse[0], coordinateur=bool(entete.get("coordinateur")))
    if cle_publique is None:
        raise RuntimeError("Aucun hôpital coordinateur : la clé publique manque.")
    return connexions, presentations, cle_publique


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", required=True, choices=list(tous_les_schemas()))
    parser.add_argument("--hopitaux", type=int, required=True, help="nombre d'hôpitaux attendus")
    parser.add_argument("--port", type=int, default=PORT_DEFAUT)
    parser.add_argument("--tableau", default=None, help="adresse du tableau de bord, ex. http://127.0.0.1:8080")
    parser.add_argument("--pause", type=float, default=0.0, help="pause (s) avant de renvoyer la somme")
    parser.add_argument("--delai", type=float, default=900.0, help="attente maximale (s) d'un hôpital")
    args = parser.parse_args()
    diffuseur = Diffuseur(args.tableau, "serveur")
    diffuseur.envoyer("serveur_demarre", schema=args.schema, nb_hopitaux=args.hopitaux, port=args.port)

    ecoute = socket.create_server(("0.0.0.0", args.port), reuse_port=False)
    print(f"Serveur {args.schema} en écoute sur le port {args.port}, "
          f"en attente de {args.hopitaux} hôpitaux...", flush=True)
    connexions, presentations, cle_publique = accueillir(ecoute, args.hopitaux, diffuseur)
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
    diffuseur.envoyer("demarrer", jeu=jeu, schema=args.schema, nb_rounds=nb_rounds, n_total=n_total,
                      empreinte=signe, hopitaux=[{"indice": k, "nom": presentations[k]["nom"],
                                                  "n_k": presentations[k]["n_k"]} for k in ordre])
    for sock in sockets:
        sock.settimeout(args.delai)

    def recevoir_et_signaler(k_sock):
        k, sock = k_sock
        recu = recevoir(sock)
        entete, blocs, octets, duree = recu
        apercu = {"hex": apercu_octets(blocs)}
        if args.schema == "sans" and blocs:  # sans chiffrement, le serveur lit les poids directement
            import numpy as np
            valeurs = np.frombuffer(blocs[0][:48], dtype=np.float64) / presentations[k]["n_k"]
            apercu = {"clair": [round(float(v), 5) for v in valeurs]}
        diffuseur.envoyer("poids_recu", round=entete.get("round"), indice=k, nom=presentations[k]["nom"],
                          octets=octets, t_transfert=duree, nb_blocs=len(blocs), **apercu)
        return recu

    historique = []
    with ThreadPoolExecutor(max_workers=len(sockets)) as pool, PicMemoire() as memoire:
        for numero in range(1, nb_rounds + 1):
            debut_round = time.perf_counter()
            # 1. Réception en parallèle des poids chiffrés de chaque hôpital
            try:
                recus = list(pool.map(recevoir_et_signaler, zip(ordre, sockets)))
            except (socket.timeout, TimeoutError, ConnectionError, OSError) as erreur:
                message = (f"Round {numero} : un hôpital ne répond plus ({type(erreur).__name__}). "
                           "Le serveur prévient les autres hôpitaux et s'arrête.")
                print("ERREUR :", message, flush=True)
                diffuseur.envoyer("erreur", round=numero, message=message)
                for sock in sockets:
                    try:
                        envoyer(sock, {"type": "FIN", "erreur": message})
                        sock.close()
                    except OSError:
                        pass
                ecoute.close()
                diffuseur.vider()
                raise SystemExit(2)
            for entete, _, _, _ in recus:
                assert entete["type"] == "POIDS" and entete["round"] == numero, entete
            messages = [schema.depuis_octets(blocs) for _, blocs, _, _ in recus]
            t_reception = time.perf_counter() - debut_round

            # 2. Addition homomorphe, sans clé secrète
            debut = time.perf_counter()
            somme = schema.additionner(messages)
            t_agregation = time.perf_counter() - debut

            diffuseur.envoyer("agregation", round=numero, t_agregation=t_agregation,
                              hex=apercu_octets(schema.vers_octets(somme)) if args.schema != "sans" else "")
            if args.pause:
                time.sleep(args.pause)

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
            diffuseur.envoyer("somme_envoyee", round=numero, octets=envois[0][0],
                              t_round_serveur=historique[-1]["t_round_serveur"])
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
    diffuseur.envoyer("fin", ram_serveur_mo=memoire.pic_mo)
    diffuseur.vider()


if __name__ == "__main__":
    main()
