"""Les trois schémas de chiffrement homomorphe comparés dans le mémoire.

Chaque schéma a exactement les mêmes méthodes que SansChiffrement (src/federe.py) :
    chiffrer(vecteur)           -> message chiffré (côté hôpital)
    additionner(messages)       -> somme chiffrée  (côté serveur, sans clé secrète)
    dechiffrer(message, taille) -> vecteur en clair (côté hôpital, clé secrète)
    taille_octets(message)      -> taille du message envoyé sur le réseau
On peut donc changer de schéma sans toucher au reste du système.

Gestion des clés (section 3.4 du rapport) :
- l'hôpital coordinateur crée la paire de clés ;
- les hôpitaux gardent la clé secrète ;
- le serveur reçoit seulement un contexte PUBLIC, sans clé secrète.
Pour que la simulation soit fidèle, chaque message CKKS/BFV est transformé
en octets (« sérialisé »), comme s'il passait par le réseau, puis relu par
le serveur avec le contexte public.

Les trois schémas sont réglés à 128 bits de sécurité (section 3.3.2).

Pour la version en réseau (src/serveur.py et src/hopital.py), chaque schéma sait
aussi :
    exporter_cles()               -> (clé secrète, clé publique) en octets
    depuis_cles(secrete, publique) -> schéma rechargé à partir de ces octets
    vers_octets(message)          -> liste de blocs d'octets à envoyer
    depuis_octets(blocs)          -> message reconstruit à la réception
Le serveur est créé avec depuis_cles(publique=...) : il ne reçoit jamais la
clé secrète et ne peut donc qu'additionner.
"""
import json
import numpy as np
import tenseal as ts
from phe import paillier

# Facteur de conversion des décimaux en entiers pour BFV et Paillier.
# Un poids de 0,12345678 devient 1 234 568 : on garde 7 chiffres après la virgule.
FACTEUR_ECHELLE = 10 ** 7


def _en_entiers(vecteur):
    return np.rint(np.asarray(vecteur, dtype=np.float64) * FACTEUR_ECHELLE).astype(np.int64)


def _en_decimaux(entiers):
    return np.asarray(entiers, dtype=np.float64) / FACTEUR_ECHELLE


def _decouper(vecteur, taille_bloc):
    """Coupe un long vecteur en blocs qui tiennent dans un seul message chiffré."""
    return [vecteur[i:i + taille_bloc] for i in range(0, len(vecteur), taille_bloc)]


# ---------------------------------------------------------------------------
# CKKS : nombres décimaux, calcul approché (TenSEAL)
# ---------------------------------------------------------------------------

class CKKS:
    """CKKS avec les paramètres de Hegde et al. (2025, p. 26023, tableau IV)."""

    nom = "CKKS"
    DEGRE = 16384                      # degré polynomial
    MODULES = [60, 40, 40, 60]         # tailles du module de coefficients (bits)
    ECHELLE = 2 ** 40                  # précision

    def __init__(self):
        # Contexte complet (avec clé secrète) : gardé par les hôpitaux.
        self.contexte_hopitaux = ts.context(ts.SCHEME_TYPE.CKKS,
                                            poly_modulus_degree=self.DEGRE,
                                            coeff_mod_bit_sizes=self.MODULES)
        self.contexte_hopitaux.global_scale = self.ECHELLE
        # Contexte public : envoyé au serveur, sans la clé secrète.
        self.contexte_serveur = self.contexte_hopitaux.copy()
        self.contexte_serveur.make_context_public()
        self.places = self.DEGRE // 2   # nombre de valeurs par message chiffré

    # --- Clés et transport (version en réseau) ---
    def exporter_cles(self):
        return (self.contexte_hopitaux.serialize(save_secret_key=True),
                self.contexte_serveur.serialize())

    @classmethod
    def depuis_cles(cls, secrete=None, publique=None):
        schema = cls.__new__(cls)
        schema.contexte_hopitaux = ts.context_from(secrete) if secrete else None
        schema.contexte_serveur = ts.context_from(publique if publique else secrete)
        if schema.contexte_hopitaux is None and schema.contexte_serveur.has_secret_key():
            raise ValueError("Le contexte du serveur ne doit pas contenir la clé secrète.")
        schema.places = cls.DEGRE // 2
        return schema

    def vers_octets(self, message):
        return list(message)

    def depuis_octets(self, blocs):
        return list(blocs)

    def chiffrer(self, vecteur):
        return [ts.ckks_vector(self.contexte_hopitaux, bloc.tolist()).serialize()
                for bloc in _decouper(np.asarray(vecteur, dtype=np.float64), self.places)]

    def additionner(self, messages):
        resultat = []
        for blocs in zip(*messages):  # le même bloc chez chaque hôpital
            somme = ts.ckks_vector_from(self.contexte_serveur, blocs[0])
            for bloc in blocs[1:]:
                somme += ts.ckks_vector_from(self.contexte_serveur, bloc)
            resultat.append(somme.serialize())
        return resultat

    def dechiffrer(self, message, taille):
        valeurs = []
        for bloc in message:
            valeurs.extend(ts.ckks_vector_from(self.contexte_hopitaux, bloc).decrypt())
        return np.asarray(valeurs[:taille], dtype=np.float64)

    def taille_octets(self, message):
        return sum(len(bloc) for bloc in message)


# ---------------------------------------------------------------------------
# BFV : nombres entiers, calcul exact (TenSEAL)
# ---------------------------------------------------------------------------

class BFV:
    """BFV avec un degré de 8192 et le module de coefficients par défaut de
    Microsoft SEAL, qui garantit 128 bits de sécurité. Le module du texte clair
    est un nombre premier d'environ 40 bits, assez grand pour contenir la somme
    des poids convertis en entiers sans débordement."""

    nom = "BFV"
    DEGRE = 8192
    MODULE_TEXTE_CLAIR = 1099511922689  # premier, égal à 1 modulo 2 x 8192

    def __init__(self):
        self.contexte_hopitaux = ts.context(ts.SCHEME_TYPE.BFV,
                                            poly_modulus_degree=self.DEGRE,
                                            plain_modulus=self.MODULE_TEXTE_CLAIR)
        self.contexte_serveur = self.contexte_hopitaux.copy()
        self.contexte_serveur.make_context_public()
        self.places = self.DEGRE

    exporter_cles = CKKS.exporter_cles
    vers_octets = CKKS.vers_octets
    depuis_octets = CKKS.depuis_octets

    @classmethod
    def depuis_cles(cls, secrete=None, publique=None):
        schema = cls.__new__(cls)
        schema.contexte_hopitaux = ts.context_from(secrete) if secrete else None
        schema.contexte_serveur = ts.context_from(publique if publique else secrete)
        if schema.contexte_hopitaux is None and schema.contexte_serveur.has_secret_key():
            raise ValueError("Le contexte du serveur ne doit pas contenir la clé secrète.")
        schema.places = cls.DEGRE
        return schema

    def chiffrer(self, vecteur):
        entiers = _en_entiers(vecteur)
        if np.abs(entiers).max() >= self.MODULE_TEXTE_CLAIR // 8:
            raise OverflowError("Valeurs trop grandes pour le module du texte clair BFV.")
        return [ts.bfv_vector(self.contexte_hopitaux, bloc.tolist()).serialize()
                for bloc in _decouper(entiers, self.places)]

    def additionner(self, messages):
        resultat = []
        for blocs in zip(*messages):
            somme = ts.bfv_vector_from(self.contexte_serveur, blocs[0])
            for bloc in blocs[1:]:
                somme += ts.bfv_vector_from(self.contexte_serveur, bloc)
            resultat.append(somme.serialize())
        return resultat

    def dechiffrer(self, message, taille):
        entiers = []
        for bloc in message:
            entiers.extend(ts.bfv_vector_from(self.contexte_hopitaux, bloc).decrypt())
        return _en_decimaux(entiers[:taille])

    def taille_octets(self, message):
        return sum(len(bloc) for bloc in message)


# ---------------------------------------------------------------------------
# Paillier : nombres entiers, addition seulement, une valeur par chiffré (phe)
# ---------------------------------------------------------------------------

class Paillier:
    """Paillier avec une clé de 3072 bits (128 bits de sécurité).
    Version de base de la bibliothèque phe : chaque poids est chiffré seul."""

    nom = "Paillier"
    TAILLE_CLE = 3072

    def __init__(self, taille_cle=None):
        self.cle_publique, self.cle_secrete = paillier.generate_paillier_keypair(
            n_length=taille_cle or self.TAILLE_CLE)
        self._preparer()

    def _preparer(self):
        self.n = self.cle_publique.n
        self.n_carre = self.cle_publique.nsquare
        # Un chiffré est un nombre modulo n², soit 2 x 3072 bits = 768 octets.
        self.octets_par_chiffre = (self.n_carre.bit_length() + 7) // 8

    # --- Clés et transport (version en réseau) ---
    def exporter_cles(self):
        secrete = json.dumps({"n": str(self.n), "p": str(self.cle_secrete.p),
                              "q": str(self.cle_secrete.q)}).encode()
        publique = json.dumps({"n": str(self.n)}).encode()
        return secrete, publique

    @classmethod
    def depuis_cles(cls, secrete=None, publique=None):
        schema = cls.__new__(cls)
        donnees = json.loads(secrete if secrete else publique)
        schema.cle_publique = paillier.PaillierPublicKey(int(donnees["n"]))
        schema.cle_secrete = (paillier.PaillierPrivateKey(schema.cle_publique, int(donnees["p"]),
                                                          int(donnees["q"])) if secrete else None)
        schema._preparer()
        return schema

    def vers_octets(self, message):
        # Chaque chiffré devient un bloc de taille fixe (768 octets pour 3072 bits).
        return [c.to_bytes(self.octets_par_chiffre, "big") for c in message]

    def depuis_octets(self, blocs):
        return [int.from_bytes(b, "big") for b in blocs]

    def chiffrer(self, vecteur):
        # On garde seulement le grand nombre entier du chiffré : c'est
        # exactement ce qui passerait sur le réseau.
        return [self.cle_publique.raw_encrypt(int(valeur) % self.n)
                for valeur in _en_entiers(vecteur)]

    def additionner(self, messages):
        # Addition homomorphe de Paillier : produit des chiffrés modulo n².
        resultat = list(messages[0])
        for message in messages[1:]:
            resultat = [(a * b) % self.n_carre for a, b in zip(resultat, message)]
        return resultat

    def dechiffrer(self, message, taille):
        entiers = []
        for chiffre in message[:taille]:
            valeur = self.cle_secrete.raw_decrypt(chiffre)
            # Les nombres négatifs sont codés dans la moitié haute de [0, n).
            entiers.append(valeur - self.n if valeur > self.n // 2 else valeur)
        return _en_decimaux(entiers)

    def taille_octets(self, message):
        return len(message) * self.octets_par_chiffre


SCHEMAS = {"ckks": CKKS, "bfv": BFV, "paillier": Paillier}