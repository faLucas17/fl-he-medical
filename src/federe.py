"""Simulation de l'apprentissage fédéré (algorithme FedAvg).

Le protocole suit la section 3.4 du rapport :
1. chaque hôpital part du modèle global et l'entraîne sur ses données ;
2. il multiplie ses poids par son nombre de patients (n_k), puis les
   « prépare » pour l'envoi (en clair ici, chiffrés plus tard) ;
3. le serveur additionne les contributions reçues ;
4. les hôpitaux récupèrent la somme, la divisent par n (nombre total de
   patients) et obtiennent le nouveau modèle global.

Le schéma utilisé est un objet « interchangeable » : ici, SansChiffrement.
Les schémas CKKS, BFV et Paillier auront exactement les mêmes méthodes.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.donnees import charger
from src.modele import (ReseauTabulaire, ecrire_poids, entrainer_localement,
                        evaluer, lire_poids)

NB_ROUNDS = 10
EPOQUES_LOCALES = 5


class SansChiffrement:
    """Schéma de référence : les poids circulent en clair (FL classique)."""

    nom = "Sans chiffrement"

    def chiffrer(self, vecteur):
        return np.asarray(vecteur, dtype=np.float64)

    def additionner(self, messages):
        return np.sum(messages, axis=0)

    def dechiffrer(self, message, taille):
        return np.asarray(message[:taille], dtype=np.float64)

    def taille_octets(self, message):
        return message.nbytes


def simuler(donnees, schema, nb_rounds=NB_ROUNDS, epoques=EPOQUES_LOCALES, graine=42):
    """Lance un apprentissage fédéré complet et renvoie les mesures de chaque round."""
    torch.manual_seed(graine)
    nb_variables = donnees.X_test.shape[1]
    modele_global = ReseauTabulaire(nb_variables)
    poids_global = lire_poids(modele_global)
    nb_poids = len(poids_global)
    tailles = [len(y) for y in donnees.y_hopitaux]
    n_total = sum(tailles)

    historique = []
    for numero in range(1, nb_rounds + 1):
        t_entrainement = t_chiffrement = 0.0
        messages = []

        # --- Côté hôpitaux : entraînement local puis chiffrement ---
        for k, (X, y) in enumerate(zip(donnees.X_hopitaux, donnees.y_hopitaux)):
            modele_local = ReseauTabulaire(nb_variables)
            ecrire_poids(modele_local, poids_global)
            debut = time.perf_counter()
            entrainer_localement(modele_local, X, y, epoques=epoques,
                                 graine=graine + 1000 * numero + k)
            t_entrainement += time.perf_counter() - debut

            debut = time.perf_counter()
            messages.append(schema.chiffrer(lire_poids(modele_local) * tailles[k]))
            t_chiffrement += time.perf_counter() - debut

        # --- Côté serveur : simple addition, sans déchiffrement ---
        debut = time.perf_counter()
        somme = schema.additionner(messages)
        t_agregation = time.perf_counter() - debut

        # --- Côté hôpital : déchiffrement puis division par n ---
        debut = time.perf_counter()
        poids_global = schema.dechiffrer(somme, nb_poids) / n_total
        t_dechiffrement = time.perf_counter() - debut

        ecrire_poids(modele_global, poids_global)
        mesures = evaluer(modele_global, donnees.X_test, donnees.y_test)
        mesures.update(round=numero, t_entrainement=t_entrainement,
                       t_chiffrement=t_chiffrement, t_agregation=t_agregation,
                       t_dechiffrement=t_dechiffrement,
                       octets_par_message=schema.taille_octets(messages[0]))
        historique.append(mesures)
    return historique, nb_poids


def reference_centralisee(donnees, nb_rounds=NB_ROUNDS, epoques=EPOQUES_LOCALES, graine=42):
    """Point de comparaison : un seul modèle entraîné sur toutes les données
    réunies (ce que le FL cherche justement à éviter), avec le même nombre
    total d'époques que le FL."""
    torch.manual_seed(graine)
    modele = ReseauTabulaire(donnees.X_test.shape[1])
    X = np.concatenate(donnees.X_hopitaux)
    y = np.concatenate(donnees.y_hopitaux)
    entrainer_localement(modele, X, y, epoques=nb_rounds * epoques, graine=graine)
    return evaluer(modele, donnees.X_test, donnees.y_test)


def enregistrer(historique, nom_jeu, schema):
    """Sauvegarde les mesures dans results/<jeu>_<schéma>.csv."""
    dossier = Path(__file__).resolve().parent.parent / "results"
    dossier.mkdir(exist_ok=True)
    nom_fichier = f"{nom_jeu}_{schema.nom.lower().replace(' ', '_')}.csv"
    pd.DataFrame(historique).to_csv(dossier / nom_fichier, index=False)
    return dossier / nom_fichier


def afficher(historique, nom_jeu, schema, nb_poids):
    print(f"\n=== {nom_jeu} | {schema.nom} | {nb_poids} poids ===")
    print("Round | Accuracy |  F1   |  AUC  | Entraîn. (s) | Chiffr. (s) | Message")
    for m in historique:
        print(f"{m['round']:>5} |  {m['accuracy']:.3f}   | {m['f1']:.3f} | {m['auc']:.3f} |"
              f"    {m['t_entrainement']:6.3f}    |   {m['t_chiffrement']:6.4f}    |"
              f" {m['octets_par_message'] / 1024:6.1f} Ko")


if __name__ == "__main__":
    jeux = sys.argv[1:] or ["pima", "heart_uci", "ckd"]
    for nom_jeu in jeux:
        donnees = charger(nom_jeu)
        schema = SansChiffrement()
        resultats, nb = simuler(donnees, schema)
        afficher(resultats, nom_jeu, schema, nb)
        centralise = reference_centralisee(donnees)
        print(f"Référence centralisée : accuracy {centralise['accuracy']:.3f} | "
              f"F1 {centralise['f1']:.3f} | AUC {centralise['auc']:.3f}")
        print("Résultats enregistrés dans", enregistrer(resultats, nom_jeu, schema))