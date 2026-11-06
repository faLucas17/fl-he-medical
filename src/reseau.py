"""Format des messages échangés sur le réseau entre les hôpitaux et le serveur.

Chaque message a deux parties :
1. un en-tête en JSON (type du message, numéro du round, etc.) ;
2. une liste de blocs d'octets (les poids chiffrés).

Sur le fil, un message s'écrit ainsi :
    [4 octets : taille de l'en-tête][en-tête JSON]
    [4 octets : nombre de blocs]
    pour chaque bloc : [8 octets : taille du bloc][bloc]

On n'utilise pas pickle : un serveur qui « dé-pickle » des données reçues du
réseau peut exécuter du code envoyé par un attaquant. Ici, le serveur ne lit
que du JSON et des octets bruts.

Types de messages :
    BONJOUR   hôpital -> serveur  (nom, n_k ; le coordinateur ajoute la clé publique)
    DEMARRER  serveur -> hôpitaux (n total, empreinte de la clé publique)
    POIDS     hôpital -> serveur  (poids chiffrés n_k x W_k d'un round)
    SOMME     serveur -> hôpitaux (somme chiffrée C du round)
    FIN       serveur -> hôpitaux
"""
import hashlib
import json
import socket
import struct
import time

PORT_DEFAUT = 5050


def envoyer(sock, entete, blocs=()):
    """Envoie un message et renvoie (nombre d'octets envoyés, durée en secondes)."""
    texte = json.dumps(entete).encode("utf-8")
    morceaux = [struct.pack("!I", len(texte)), texte, struct.pack("!I", len(blocs))]
    for bloc in blocs:
        morceaux.append(struct.pack("!Q", len(bloc)))
        morceaux.append(bytes(bloc))
    donnees = b"".join(morceaux)
    debut = time.perf_counter()
    sock.sendall(donnees)
    return len(donnees), time.perf_counter() - debut


def _lire_exactement(sock, n):
    tampon = bytearray()
    while len(tampon) < n:
        morceau = sock.recv(min(n - len(tampon), 1 << 20))
        if not morceau:
            raise ConnectionError("La connexion a été fermée par l'autre machine.")
        tampon.extend(morceau)
    return bytes(tampon)


def recevoir(sock):
    """Reçoit un message.

    Renvoie (en-tête, blocs, octets reçus, durée du transfert). La durée part
    de l'arrivée des premiers octets : elle ne compte pas l'attente avant que
    l'autre machine commence à envoyer.
    """
    debut_entete = _lire_exactement(sock, 4)
    debut = time.perf_counter()
    taille_entete = struct.unpack("!I", debut_entete)[0]
    entete = json.loads(_lire_exactement(sock, taille_entete).decode("utf-8"))
    nb_blocs = struct.unpack("!I", _lire_exactement(sock, 4))[0]
    blocs, total = [], 8 + taille_entete
    for _ in range(nb_blocs):
        taille = struct.unpack("!Q", _lire_exactement(sock, 8))[0]
        blocs.append(_lire_exactement(sock, taille))
        total += 8 + taille
    return entete, blocs, total, time.perf_counter() - debut


def empreinte(cle_publique):
    """Empreinte courte de la clé publique : tous les participants doivent
    utiliser la même paire de clés (contrôle contre une clé périmée)."""
    return hashlib.sha256(cle_publique).hexdigest()[:16]


def connecter(adresse, essais=60):
    """Se connecte au serveur « ip:port », en réessayant pendant une minute."""
    hote, _, port = adresse.rpartition(":")
    for _ in range(essais):
        try:
            sock = socket.create_connection((hote or "127.0.0.1", int(port)))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            time.sleep(1)
    raise ConnectionError(f"Impossible de joindre le serveur {adresse}.")
