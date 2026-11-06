"""Tests automatiques du système (simulation et réseau).

Lancement :  python -m tests.test_systeme
Durée : environ 1 minute. Paillier utilise ici une petite clé de 1024 bits
pour aller vite ; les expériences du mémoire utilisent 3072 bits.
"""
import socket
import threading

import numpy as np
import tenseal as ts
import torch

from src.chiffrement import BFV, CKKS, Paillier
from src.federe import SansChiffrement
from src.modele import ReseauTabulaire, ecrire_poids, lire_poids
from src.reseau import envoyer, recevoir

reussis = 0


def verifier(condition, message):
    global reussis
    if not condition:
        raise AssertionError("ÉCHEC : " + message)
    reussis += 1
    print("  OK  ", message)


def petit_paillier():
    return Paillier(taille_cle=1024)


def test_agregation_et_cles():
    """Chaque schéma : chiffrer, passer en octets, additionner côté serveur
    (clé publique seule), déchiffrer côté hôpital."""
    print("\n1. Agrégation chiffrée avec un serveur qui n'a que la clé publique")
    rng = np.random.default_rng(0)
    poids = [rng.normal(0, 0.5, 1409) * n for n in (242, 235, 160)]
    attendu = np.sum(poids, axis=0)
    for fabrique in (SansChiffrement, CKKS, BFV, petit_paillier):
        hopital = fabrique()
        secrete, publique = hopital.exporter_cles()
        classe = type(hopital)
        serveur = classe.depuis_cles(publique=publique)
        autre_hopital = classe.depuis_cles(secrete=secrete)
        # hôpital -> octets -> serveur
        messages = [serveur.depuis_octets(hopital.vers_octets(hopital.chiffrer(p))) for p in poids]
        somme = serveur.additionner(messages)
        # serveur -> octets -> un AUTRE hôpital, qui a seulement relu les clés
        resultat = autre_hopital.dechiffrer(autre_hopital.depuis_octets(serveur.vers_octets(somme)), 1409)
        tolerance = 1e-3 if classe is CKKS else 1e-4
        verifier(np.allclose(resultat, attendu, atol=tolerance),
                 f"{classe.nom} : somme déchiffrée = somme en clair (écart max {np.abs(resultat - attendu).max():.1e})")


def test_serveur_sans_cle_secrete():
    print("\n2. Le serveur ne peut pas déchiffrer")
    for fabrique in (CKKS, BFV, petit_paillier):
        hopital = fabrique()
        _, publique = hopital.exporter_cles()
        serveur = type(hopital).depuis_cles(publique=publique)
        message = hopital.chiffrer(np.ones(10))
        try:
            if isinstance(serveur, Paillier):
                peut_dechiffrer = serveur.cle_secrete is not None
            else:  # on essaie de déchiffrer avec le contexte du serveur lui-même
                lire = ts.ckks_vector_from if isinstance(serveur, CKKS) else ts.bfv_vector_from
                lire(serveur.contexte_serveur, message[0]).decrypt()
                peut_dechiffrer = True
        except Exception:
            peut_dechiffrer = False
        verifier(not peut_dechiffrer, f"{type(hopital).nom} : le déchiffrement échoue sans clé secrète")


def test_copie_des_poids():
    print("\n3. Le modèle ne partage pas sa mémoire avec le vecteur de poids")
    torch.manual_seed(0)
    modele = ReseauTabulaire(8)
    vecteur = lire_poids(modele)
    avant = vecteur.copy()
    autre = ReseauTabulaire(8)
    ecrire_poids(autre, vecteur)
    with torch.no_grad():
        for p in autre.parameters():
            p.add_(1.0)
    verifier(np.array_equal(vecteur, avant), "modifier le modèle local ne change pas le modèle global")


def test_transport_reseau():
    print("\n4. Envoi et réception d'un message par le réseau")
    blocs = [bytes(range(256)) * 3, b"", b"x" * 700_000]
    ecoute = socket.create_server(("127.0.0.1", 0))
    port = ecoute.getsockname()[1]
    recu = {}

    def serveur():
        sock, _ = ecoute.accept()
        recu["message"] = recevoir(sock)
        sock.close()

    fil = threading.Thread(target=serveur)
    fil.start()
    client = socket.create_connection(("127.0.0.1", port))
    octets, _ = envoyer(client, {"type": "POIDS", "round": 3}, blocs)
    fil.join()
    client.close()
    ecoute.close()
    entete, blocs_recus, octets_recus, _ = recu["message"]
    verifier(entete == {"type": "POIDS", "round": 3}, "en-tête reçu intact")
    verifier(blocs_recus == blocs, "blocs reçus identiques aux blocs envoyés")
    verifier(octets == octets_recus, f"même nombre d'octets des deux côtés ({octets})")


if __name__ == "__main__":
    test_agregation_et_cles()
    test_serveur_sans_cle_secrete()
    test_copie_des_poids()
    test_transport_reseau()
    print(f"\nTous les tests sont réussis ({reussis} vérifications).")
