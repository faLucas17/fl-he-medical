"""Attaque par inversion de gradients (hypothèse H4), avec et sans chiffrement.

Scénario (modèle de menace du chapitre 3) : le serveur est honnête-mais-curieux.
Il suit le protocole, mais il essaie de reconstruire les données d'un patient
à partir du message qu'un hôpital lui envoie.

Pour se placer dans le cas le plus favorable à l'attaquant, comme Zhu et al.
(« Deep Leakage from Gradients »), l'hôpital fait UN pas d'apprentissage sur un
petit lot de B patients, puis envoie ses poids W_k = W - η·g. L'attaquant :
- connaît le modèle global W, le taux η et les étiquettes des patients ;
- sans chiffrement, il retrouve le gradient g = (W - W_k) / η, puis cherche des
  patients fictifs x̂ dont le gradient ressemble le plus possible à g
  (optimisation L-BFGS, formule (1.5) du mémoire) ;
- avec chiffrement, il ne reçoit qu'un message chiffré. Sa meilleure tentative
  est de le « déchiffrer » avec une clé qu'il a créée lui-même (il n'a pas la
  vraie clé secrète), puis de lancer la même attaque sur ce qu'il obtient.

L'attaque est réussie quand le patient reconstruit est plus proche du vrai
patient que de tous les autres patients de l'hôpital : c'est une
RÉ-IDENTIFICATION. La configuration « hasard » sert de référence : elle
« reconstruit » des patients tirés au hasard et donne le taux obtenu par chance.

Utilisation :
    python -m src.attaque                         # Pima, tous les cas (environ 5 minutes)
    python -m src.attaque --jeux pima heart_uci ckd
    python -m src.attaque --rapide                # moins d'essais, pour tester

Résultats :
    results/attaque/essais.csv     une ligne par essai
    results/attaque/resume.csv     taux de ré-identification et erreur moyenne
    results/attaque/exemple.csv    un patient : vrai, reconstruit en clair, reconstruit chiffré
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from src.chiffrement import BFV, CKKS, Paillier
from src.donnees import charger
from src.modele import ReseauTabulaire, lire_poids

DOSSIER = Path(__file__).resolve().parent.parent / "results" / "attaque"
TAUX = 0.01          # taux d'apprentissage η du pas envoyé par l'hôpital
ESSAIS_LBFGS = 4     # départs aléatoires de l'attaquant (il garde le meilleur)
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Outils : modèle, gradient d'un lot, attaque
# ---------------------------------------------------------------------------
def modele_global(nb_variables, graine=42):
    """Le modèle global du round 1 (même départ que les expériences)."""
    torch.manual_seed(graine)
    modele = ReseauTabulaire(nb_variables).double()
    return modele, torch.tensor(lire_poids(modele), dtype=torch.float64)


def gradient(modele, X, y, creer_graphe=False):
    perte = nn.BCEWithLogitsLoss()(modele(X), y)
    return torch.autograd.grad(perte, list(modele.parameters()), create_graph=creer_graphe)


def aplatir(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def inverser(modele, g_cible, y, nb_variables, graine):
    """Cherche B patients fictifs dont le gradient ressemble à g_cible (formule (1.5))."""
    B = len(y)
    meilleur, meilleure_distance = None, float("inf")
    echelle = g_cible.norm() ** 2 + 1e-12
    for depart in range(ESSAIS_LBFGS):
        torch.manual_seed(graine * 100 + depart)
        x = torch.randn(B, nb_variables, requires_grad=True)
        optimiseur = torch.optim.LBFGS([x], lr=1, max_iter=30, history_size=50,
                                       line_search_fn="strong_wolfe")

        def cout():
            optimiseur.zero_grad()
            d = ((aplatir(gradient(modele, x, y, True)) - g_cible) ** 2).sum() / echelle
            d.backward()
            return d

        for _ in range(15):
            try:
                optimiseur.step(cout)
            except RuntimeError:
                break
        x = x.detach()
        distance = (float(((aplatir(gradient(modele, x, y)) - g_cible) ** 2).sum() / echelle)
                    if torch.isfinite(x).all() else float("inf"))
        if distance < meilleure_distance:
            meilleur, meilleure_distance = x.detach().clone(), distance
    if meilleur is None:  # l'optimisation a divergé à chaque départ
        meilleur = torch.zeros(B, nb_variables)
    return meilleur, meilleure_distance


def _signe_paillier(valeur, n):
    return valeur - n if valeur > n // 2 else valeur


def _vers_reel(entier):
    """Entier immense -> réel (borné à ±1e300 pour éviter un dépassement)."""
    if abs(entier) > 10 ** 307:
        return 1e300 if entier > 0 else -1e300
    return entier / 10 ** 7


def evaluer_reconstruction(X_vrais, X_rec, X_hopital, indices_vrais):
    """Erreur moyenne (en écarts-types) et taux de ré-identification."""
    X_rec = np.nan_to_num(X_rec, nan=0.0, posinf=1e6, neginf=-1e6)
    couts = ((X_vrais[:, None, :] - X_rec[None, :, :]) ** 2).sum(-1)
    lignes, colonnes = linear_sum_assignment(couts)          # associe chaque reconstruit à un vrai patient
    erreur = np.abs(X_vrais[lignes] - X_rec[colonnes]).mean()
    reidentifies = 0
    for i, j in zip(lignes, colonnes):
        plus_proche = np.argmin(((X_hopital - X_rec[j]) ** 2).sum(1))
        reidentifies += int(plus_proche == indices_vrais[i])
    return erreur, reidentifies / len(lignes)


# ---------------------------------------------------------------------------
# Expérience
# ---------------------------------------------------------------------------
def un_essai(jeu, donnees, config, B, essai, schemas):
    X_hop = donnees.X_hopitaux[0].astype(np.float64)
    y_hop = donnees.y_hopitaux[0].astype(np.float64)
    n_k = len(y_hop)
    d = X_hop.shape[1]
    rng = np.random.default_rng(1000 * B + essai)
    indices = rng.choice(n_k, size=B, replace=False)
    X = torch.tensor(X_hop[indices])
    y = torch.tensor(y_hop[indices])

    # --- Côté hôpital : un pas d'apprentissage, puis le message ---
    modele, W = modele_global(d)
    g_vrai = aplatir(gradient(modele, X, y)).detach()
    W_k = W - TAUX * g_vrai
    message_clair = (W_k * n_k).numpy()

    # --- Côté serveur : ce qu'il obtient ---
    if config == "hasard":  # référence : on « reconstruit » des patients tirés au hasard
        X_rec = torch.tensor(rng.normal(size=(B, d)))
        erreur, reid = evaluer_reconstruction(X_hop[indices], X_rec.numpy(), X_hop, indices)
        return {"jeu": jeu, "configuration": config, "taille_lot": B, "essai": essai,
                "erreur_moyenne": erreur, "reidentification": reid, "distance_gradient": float("nan"),
                "duree_s": 0.0, "nb_patients_hopital": n_k}, (indices, X_rec.numpy())
    if config == "sans":
        recu = message_clair
    elif config == "paillier":
        chiffre = schemas[config].chiffrer(message_clair)
        attaquant = Paillier()  # sa propre clé
        recu = np.array([_vers_reel(_signe_paillier(attaquant.cle_secrete.raw_decrypt(c), attaquant.n))
                         for c in chiffre], dtype=np.float64)
    else:
        hopital = schemas[config]
        chiffre = hopital.chiffrer(message_clair)
        attaquant = {"ckks": CKKS, "bfv": BFV}[config]()  # sa propre clé
        recu = np.asarray(attaquant.dechiffrer(chiffre, len(message_clair)), dtype=np.float64)
    W_k_vu = torch.tensor(recu / n_k)
    g_vu = (W - W_k_vu) / TAUX
    g_vu = torch.nan_to_num(g_vu, nan=0.0, posinf=1e30, neginf=-1e30)

    debut = time.perf_counter()
    X_rec, distance = inverser(modele, g_vu, y, d, graine=1000 * B + essai)
    duree = time.perf_counter() - debut
    erreur, reid = evaluer_reconstruction(X_hop[indices], X_rec.numpy(), X_hop, indices)
    return {"jeu": jeu, "configuration": config, "taille_lot": B, "essai": essai,
            "erreur_moyenne": erreur, "reidentification": reid, "distance_gradient": distance,
            "duree_s": duree, "nb_patients_hopital": n_k}, (indices, X_rec.numpy())


def exemple(jeu, donnees, schemas):
    """Un patient (B = 1) : valeurs réelles, reconstruites en clair et après chiffrement CKKS."""
    lignes = []
    resultats = {}
    for config in ("sans", "ckks"):
        _, (indices, X_rec) = un_essai(jeu, donnees, config, 1, 0, schemas)
        resultats[config] = X_rec[0]
    vrai = donnees.X_hopitaux[0][indices[0]].astype(np.float64)
    unites = {}
    if donnees.standardiseur is not None:  # valeurs dans les unités d'origine (mg/dL, années...)
        for cle, v in (("vrai", vrai), ("sans", resultats["sans"]), ("ckks", resultats["ckks"])):
            unites[cle] = donnees.standardiseur.inverse_transform(np.nan_to_num(v).reshape(1, -1))[0]
    for k, nom in enumerate(donnees.noms_variables):
        ligne = {"jeu": jeu, "patient": int(indices[0]), "variable": nom, "vrai": vrai[k],
                 "reconstruit_sans_chiffrement": resultats["sans"][k],
                 "reconstruit_avec_ckks": resultats["ckks"][k]}
        if unites:
            ligne.update({"vrai_unite": unites["vrai"][k], "sans_chiffrement_unite": unites["sans"][k],
                          "ckks_unite": unites["ckks"][k]})
        lignes.append(ligne)
    return lignes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jeux", nargs="+", default=["pima"], choices=["pima", "heart_uci", "ckd"])
    parser.add_argument("--rapide", action="store_true", help="moins d'essais (test)")
    args = parser.parse_args()

    nb = 3 if args.rapide else 10
    plan = [("sans", B, nb) for B in (1, 2, 4, 8, 16)]
    plan += [("hasard", B, 50) for B in (1, 2, 4, 8, 16)]
    plan += [("ckks", B, nb) for B in (1, 8)] + [("bfv", B, nb) for B in (1, 8)]
    plan += [("paillier", 1, 2 if args.rapide else 3)]   # Paillier : ~30 s de chiffrement par essai

    DOSSIER.mkdir(parents=True, exist_ok=True)
    print("Création des clés des hôpitaux (CKKS, BFV, Paillier 3072 bits)...", flush=True)
    schemas = {"ckks": CKKS(), "bfv": BFV(), "paillier": Paillier()}
    essais, exemples = [], []
    for jeu in args.jeux:
        donnees = charger(jeu)
        print(f"\n=== {jeu} : {len(donnees.y_hopitaux[0])} patients dans l'hôpital attaqué, "
              f"{donnees.X_hopitaux[0].shape[1]} variables ===", flush=True)
        for config, B, n in plan:
            ligne_essais = [un_essai(jeu, donnees, config, B, e, schemas)[0] for e in range(n)]
            essais += ligne_essais
            df = pd.DataFrame(ligne_essais)
            print(f"  {config:<9} lot de {B:>2} patient(s) : ré-identification "
                  f"{100 * df['reidentification'].mean():5.1f} % | erreur moyenne "
                  f"{df['erreur_moyenne'].mean():9.3g} écart-type", flush=True)
        exemples += exemple(jeu, donnees, schemas)

    df = pd.DataFrame(essais)
    df.to_csv(DOSSIER / "essais.csv", index=False)
    resume = df.groupby(["jeu", "configuration", "taille_lot"], sort=False).agg(
        nb_essais=("essai", "count"),
        reidentification=("reidentification", "mean"),
        erreur_moyenne=("erreur_moyenne", "median"),
        nb_patients_hopital=("nb_patients_hopital", "first")).reset_index()
    resume.to_csv(DOSSIER / "resume.csv", index=False)
    pd.DataFrame(exemples).to_csv(DOSSIER / "exemple.csv", index=False)

    print("\n=== Résumé : part des patients ré-identifiés ===")
    for _, r in resume.iterrows():
        print(f"{r['jeu']:<10} {r['configuration']:<9} lot {r['taille_lot']:>2} | "
              f"{100 * r['reidentification']:5.1f} % | "
              f"erreur médiane {r['erreur_moyenne']:.3g}")
    print("\nRésultats enregistrés dans", DOSSIER)


if __name__ == "__main__":
    main()
