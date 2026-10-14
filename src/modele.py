"""Modèle de classification et outils pour manipuler ses poids.

Le modèle est un petit réseau de neurones (perceptron multicouche) à deux
couches cachées. Il est volontairement petit : chaque poids devra être chiffré
à chaque round, et Paillier chiffre les poids un par un.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch import nn

TAILLES_CACHEES = (32, 16)


class ReseauTabulaire(nn.Module):
    """Réseau à deux couches cachées : entrée -> 32 -> 16 -> 1 sortie."""

    def __init__(self, nb_variables: int):
        super().__init__()
        c1, c2 = TAILLES_CACHEES
        self.couches = nn.Sequential(
            nn.Linear(nb_variables, c1), nn.ReLU(),
            nn.Linear(c1, c2), nn.ReLU(),
            nn.Linear(c2, 1),
        )

    def forward(self, x):
        return self.couches(x).squeeze(-1)  # un score par patient


def lire_poids(modele: nn.Module) -> np.ndarray:
    """Met tous les poids du modèle dans un seul vecteur (aplatissement)."""
    return torch.nn.utils.parameters_to_vector(modele.parameters()).detach().numpy().copy()


def ecrire_poids(modele: nn.Module, vecteur: np.ndarray) -> None:
    """Remet un vecteur de poids dans le modèle."""
    tenseur = torch.as_tensor(np.asarray(vecteur, dtype=np.float32))
    torch.nn.utils.vector_to_parameters(tenseur, modele.parameters())


def entrainer_localement(modele, X, y, epoques=5, taux=1e-3, taille_lot=32, graine=0):
    """Entraîne le modèle sur les données d'un hôpital (Adam, taux de 0,001)."""
    generateur = torch.Generator().manual_seed(graine)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=taux)
    perte = nn.BCEWithLogitsLoss()
    modele.train()
    for _ in range(epoques):
        ordre = torch.randperm(len(X_t), generator=generateur)
        for debut in range(0, len(X_t), taille_lot):
            lot = ordre[debut:debut + taille_lot]
            optimiseur.zero_grad()
            perte(modele(X_t[lot]), y_t[lot]).backward()
            optimiseur.step()


@torch.no_grad()
def evaluer(modele, X, y) -> dict:
    """Calcule les métriques du modèle sur un jeu de données."""
    modele.eval()
    probabilites = torch.sigmoid(modele(torch.tensor(X, dtype=torch.float32))).numpy()
    predictions = (probabilites >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "rappel": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "auc": roc_auc_score(y, probabilites),
    }