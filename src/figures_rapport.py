"""Figures du rapport en noir et blanc (style croquis scientifique).

Utilisation : python -m src.figures_rapport
Les images sont créées dans le dossier figures_rapport/ à la racine du projet.
Les graphiques du chapitre 5 sont calculés à partir des fichiers results/*.csv.
Les figures en couleur de la présentation restent produites par src/figures.py.
"""
import logging
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

RACINE = Path(__file__).resolve().parent.parent
OUT = RACINE / "figures_rapport"
OUT.mkdir(parents=True, exist_ok=True)
RESULTATS = RACINE / "results"

plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["hatch.linewidth"] = 0.8

NOIR, GRIS, GRIS_CLAIR, BLANC = "black", "#555555", "#E6E6E6", "white"
T = 13  # taille de texte par défaut


# ---------------------------------------------------------------------------
# Outils de dessin
# ---------------------------------------------------------------------------
def figure(l, h):
    fig, ax = plt.subplots(figsize=(l, h))
    ax.set_xlim(0, l)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def sauver(fig, nom):
    fig.savefig(OUT / nom, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boite(ax, x, y, w, h, texte="", taille=T, gras=False, fond=BLANC, tirets=False, lw=1.3, rayon=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rayon}",
                                facecolor=fond, edgecolor=NOIR, linewidth=lw,
                                linestyle=(0, (5, 3)) if tirets else "-", zorder=2))
    if texte:
        ax.text(x + w / 2, y + h / 2, texte, ha="center", va="center", fontsize=taille,
                fontweight="bold" if gras else "normal", zorder=3, linespacing=1.35)


def fleche(ax, x1, y1, x2, y2, texte=None, tx=None, ty=None, taille=T - 1, tirets=False, lw=1.3,
           ha="center", fond=True, double=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle="<|-|>" if double else "-|>", color=NOIR, lw=lw,
                                linestyle=(0, (5, 3)) if tirets else "-", shrinkA=0, shrinkB=0,
                                mutation_scale=15))
    if texte:
        ax.text(tx if tx is not None else (x1 + x2) / 2, ty if ty is not None else (y1 + y2) / 2, texte,
                ha=ha, va="center", fontsize=taille, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5) if fond else None)


def base_de_donnees(ax, x, y, w=0.9, h=0.75, texte=""):
    """Cylindre (symbole de base de données)."""
    e = h * 0.22
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, facecolor=BLANC, edgecolor="none", zorder=2))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=NOIR, lw=1.2, zorder=3)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=NOIR, lw=1.2, zorder=3)
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, facecolor=BLANC, edgecolor=NOIR, lw=1.2, zorder=2))
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, facecolor=BLANC, edgecolor=NOIR, lw=1.2, zorder=3))
    if texte:
        ax.text(x + w / 2, y - 0.2, texte, ha="center", va="top", fontsize=T - 2)


def cadenas(ax, x, y, s=0.22):
    """Petit cadenas : symbole d'une donnée chiffrée."""
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s * 0.8, facecolor=NOIR, edgecolor=NOIR, zorder=5))
    arc = matplotlib.patches.Arc((x, y + s * 0.3), s * 0.7, s * 0.8, theta1=0, theta2=180, color=NOIR, lw=1.5,
                                 zorder=5)
    ax.add_patch(arc)


def acteur(ax, x, y, nom):
    ax.add_patch(Circle((x, y + 0.75), 0.17, facecolor=BLANC, edgecolor=NOIR, lw=1.5))
    ax.plot([x, x], [y + 0.58, y + 0.15], color=NOIR, lw=1.5)
    ax.plot([x - 0.3, x + 0.3], [y + 0.45, y + 0.45], color=NOIR, lw=1.5)
    ax.plot([x, x - 0.25], [y + 0.15, y - 0.2], color=NOIR, lw=1.5)
    ax.plot([x, x + 0.25], [y + 0.15, y - 0.2], color=NOIR, lw=1.5)
    ax.text(x, y - 0.45, nom, fontsize=T - 1, ha="center", va="center")


# ---------------------------------------------------------------------------
# Chapitre 1
# ---------------------------------------------------------------------------
def fig1_1_principe_fl():
    fig, ax = figure(11, 6.2)
    boite(ax, 0.6, 4.6, 9.8, 1.2)
    ax.text(5.5, 5.4, "Serveur d'agrégation", ha="center", fontsize=T + 1, fontweight="bold")
    ax.text(5.5, 4.9, r"$W = \sum_{k=1}^{K} \frac{n_k}{n}\, W_k$", ha="center", fontsize=T + 2)
    for i, x in enumerate([0.6, 4.3, 8.0]):
        cx = x + 1.2
        boite(ax, x, 1.2, 2.4, 1.3, f"Hôpital {i + 1}\nentraînement local", taille=T - 1)
        base_de_donnees(ax, cx - 0.45, 0.0, 0.9, 0.85)
        ax.text(cx, -0.12, f"données $D_{i + 1}$", ha="center", va="top", fontsize=T - 2)
        ax.plot([cx, cx], [0.85, 1.2], color=NOIR, lw=1.2)
        fleche(ax, cx - 0.35, 2.5, cx - 0.35, 4.6, f"$W_{i + 1}$", tx=cx - 0.75, ty=3.55)
        fleche(ax, cx + 0.35, 4.6, cx + 0.35, 2.5, "$W$", tx=cx + 0.7, ty=3.55, tirets=True)
    ax.text(5.5, -0.75, "Trait plein : poids locaux envoyés. Pointillés : modèle global renvoyé. "
            "Les données restent dans chaque hôpital.", ha="center", fontsize=T - 2, style="italic")
    sauver(fig, "fig1_1_principe_apprentissage_federe.png")


def fig1_2_inversion_gradients():
    fig, ax = figure(12, 4.2)
    boite(ax, 0.1, 1.4, 2.5, 1.4, "Hôpital\ndonnées du patient $x$")
    boite(ax, 3.4, 1.4, 2.3, 1.4, "Gradients\n" + r"$\nabla W = \frac{\partial \ell(x,y)}{\partial W}$")
    boite(ax, 6.6, 1.4, 2.8, 1.4, "Attaquant\n" + r"$\min_{x'} \|\nabla W' - \nabla W\|^2$")
    boite(ax, 10.2, 1.4, 1.7, 1.4, "Donnée\nreconstruite\n$x' \\approx x$")
    fleche(ax, 2.6, 2.1, 3.4, 2.1)
    fleche(ax, 5.7, 2.1, 6.6, 2.1, "interception\nou serveur curieux", ty=3.25)
    fleche(ax, 9.4, 2.1, 10.2, 2.1)
    ax.text(6.0, 0.6, "Sans chiffrement, les gradients envoyés suffisent pour retrouver une partie des données.",
            ha="center", fontsize=T - 1, style="italic")
    sauver(fig, "fig1_2_attaque_inversion_gradients.png")


def fig1_3_techniques_protection():
    fig, ax = figure(12, 5.0)
    boite(ax, 3.6, 4.0, 4.8, 0.9, "Protection de la vie privée en apprentissage fédéré", gras=True, taille=T - 1)
    enfants = [(0.2, "Confidentialité différentielle", "ajoute un bruit aux poids"),
               (4.2, "Calcul multipartite sécurisé", "partage les poids en secrets"),
               (8.2, "Chiffrement homomorphe", "calcule sur des poids chiffrés")]
    for x, titre, desc in enfants:
        fond = GRIS_CLAIR if "homomorphe" in titre else BLANC
        boite(ax, x, 2.2, 3.6, 1.2, "", fond=fond)
        ax.text(x + 1.8, 3.0, titre, ha="center", va="center", fontsize=T - 1, fontweight="bold")
        ax.text(x + 1.8, 2.55, desc, ha="center", va="center", fontsize=T - 2, style="italic")
        ax.plot([6.0, 6.0, x + 1.8, x + 1.8], [4.0, 3.7, 3.7, 3.4], color=NOIR, lw=1.2)
    boite(ax, 6.9, 0.2, 2.4, 0.9, "Additif seul :\nPaillier", taille=T - 2)
    boite(ax, 9.5, 0.2, 2.4, 0.9, "Addition et produit :\nBFV, CKKS", taille=T - 2)
    ax.plot([10.0, 10.0], [2.2, 1.5], color=NOIR, lw=1.2)
    ax.plot([8.1, 8.1, 10.7, 10.7], [1.1, 1.5, 1.5, 1.1], color=NOIR, lw=1.2)
    sauver(fig, "fig1_3_techniques_protection.png")


def fig1_4_principe_chiffrement_homomorphe():
    fig, ax = figure(12, 4.6)
    ax.add_patch(Rectangle((4.15, 0.35), 3.7, 3.9, facecolor="none", edgecolor=NOIR, lw=1.0,
                           linestyle=(0, (5, 3))))
    ax.text(6.0, 4.0, "Serveur (sans clé secrète)", ha="center", fontsize=T - 1, style="italic")
    boite(ax, 0.2, 2.6, 1.2, 0.8, "$m_1$")
    boite(ax, 0.2, 0.9, 1.2, 0.8, "$m_2$")
    boite(ax, 2.2, 2.6, 1.5, 0.8, "$c_1$")
    boite(ax, 2.2, 0.9, 1.5, 0.8, "$c_2$")
    fleche(ax, 1.4, 3.0, 2.2, 3.0, r"$\mathrm{Enc}_{pk}$", ty=3.55, fond=False)
    fleche(ax, 1.4, 1.3, 2.2, 1.3, r"$\mathrm{Enc}_{pk}$", ty=1.85, fond=False)
    cadenas(ax, 3.55, 3.3)
    cadenas(ax, 3.55, 1.6)
    boite(ax, 4.7, 1.55, 2.6, 1.1, r"$c = c_1 \oplus c_2$")
    fleche(ax, 3.7, 3.0, 4.7, 2.3)
    fleche(ax, 3.7, 1.3, 4.7, 1.9)
    boite(ax, 8.4, 1.6, 1.4, 1.0, "$c$")
    cadenas(ax, 9.65, 2.45)
    fleche(ax, 7.3, 2.1, 8.4, 2.1)
    boite(ax, 10.6, 1.6, 1.3, 1.0, "$m_1 + m_2$")
    fleche(ax, 9.8, 2.1, 10.6, 2.1, r"$\mathrm{Dec}_{sk}$", ty=2.65, fond=False)
    sauver(fig, "fig1_4_principe_chiffrement_homomorphe.png")


# ---------------------------------------------------------------------------
# Chapitre 2
# ---------------------------------------------------------------------------
def fig2_1_cas_utilisation():
    fig, ax = figure(12, 7.2)
    boite(ax, 2.55, 0.25, 6.9, 6.7, rayon=0.03, fond="none")
    ax.text(6.0, 6.62, "Système d'apprentissage fédéré sécurisé", fontsize=T + 1, ha="center", fontweight="bold")
    cas = ["Générer les clés (pk, sk)", "Entraîner le modèle localement", "Chiffrer les paramètres",
           "Agréger les paramètres chiffrés", "Déchiffrer le modèle global", "Évaluer le modèle",
           "Comparer CKKS, BFV et Paillier", "Simuler une attaque par inversion"]
    ys = [6.0, 5.2, 4.4, 3.6, 2.8, 2.0, 1.2, 0.55]
    for y, t in zip(ys, cas):
        ax.add_patch(Ellipse((6.0, y), 5.4, 0.55, facecolor=BLANC, edgecolor=NOIR, lw=1.2))
        ax.text(6.0, y, t, fontsize=T - 1, ha="center", va="center")
    acteur(ax, 1.0, 5.4, "Hôpital\ncoordinateur")
    acteur(ax, 1.0, 3.3, "Hôpital\n(client)")
    acteur(ax, 11.0, 3.3, "Serveur\nd'agrégation")
    acteur(ax, 11.0, 0.9, "Expérimentatrice")
    lien = dict(color=NOIR, lw=0.9)
    ax.plot([1.3, 3.3], [5.75, 6.0], **lien)
    for y in [5.2, 4.4, 2.8, 2.0]:
        ax.plot([1.3, 3.3], [3.65, y], **lien)
    ax.plot([10.7, 8.7], [3.65, 3.6], **lien)
    for y in [1.2, 0.55]:
        ax.plot([10.7, 8.7], [1.25, y], **lien)
    sauver(fig, "fig2_1_cas_utilisation.png")


def fig2_2_sequence():
    fig, ax = figure(12, 7.4)
    acteurs = {"Hôpital coordinateur": 1.6, "Hôpital k (k = 1 … K)": 6.0, "Serveur d'agrégation": 10.25}
    for nom, x in acteurs.items():
        boite(ax, x - 1.45, 6.75, 2.9, 0.55, nom, taille=T - 1, gras=True, rayon=0.02)
        ax.plot([x, x], [0.2, 6.75], color=NOIR, lw=0.9, ls=(0, (5, 4)), zorder=1)
    xc, xh, xs = acteurs.values()

    def message(x1, x2, y, texte, tirets=False):
        fleche(ax, x1, y, x2, y, texte, ty=y + 0.18, tirets=tirets)

    def action(x, y, texte, w=3.1):
        boite(ax, x - w / 2, y - 0.2, w, 0.4, texte, taille=T - 2, rayon=0.03)

    action(xc, 6.35, "génère (pk, sk)", w=2.2)
    message(xc, xh, 5.85, "sk (canal sécurisé)")
    message(xc, xs, 5.35, "pk (contexte public)")
    ax.add_patch(Rectangle((0.1, 0.3), 11.85, 4.65, facecolor="none", edgecolor=NOIR, lw=1.1))
    boite(ax, 0.1, 4.6, 2.65, 0.35, "loop : t = 1 … 10", taille=T - 2, gras=True, rayon=0.0)
    action(xh, 4.25, "1. entraînement local (5 époques)")
    action(xh, 3.7, r"2. $c_k = \mathrm{Enc}_{pk}(n_k W_k)$")
    message(xh, xs, 3.2, r"3. $c_k,\ n_k$")
    action(xs, 2.65, r"4. $C = c_1 \oplus \cdots \oplus c_K$")
    message(xs, xh, 2.1, r"5. $C,\ n$", tirets=True)
    action(xh, 1.5, r"6. $W = \mathrm{Dec}_{sk}(C)\,/\,n$")
    action(xh, 0.85, "7. évaluation sur le jeu de test")
    sauver(fig, "fig2_2_diagramme_sequence.png")


# ---------------------------------------------------------------------------
# Chapitre 3
# ---------------------------------------------------------------------------
def fig3_1_modele_menace():
    fig, ax = figure(12, 5.6)
    for i, y in enumerate([3.7, 2.1, 0.5]):
        boite(ax, 0.2, y, 2.6, 1.1, f"Hôpital {i + 1}\nhonnête", taille=T - 1)
        fleche(ax, 2.8, y + 0.55, 8.6, 2.65 + (1 - i) * 0.25)
        cadenas(ax, 4.0, y + 0.55 + (2.65 + (1 - i) * 0.25 - y - 0.55) * (1.2 / 5.8))
    boite(ax, 8.6, 1.6, 3.2, 2.2, "Serveur\nhonnête-mais-curieux\n\nvoit seulement des\nmessages chiffrés",
          taille=T - 1)
    boite(ax, 4.9, 4.4, 3.0, 1.0, "Adversaire externe\n(écoute du réseau)", taille=T - 1, tirets=True)
    fleche(ax, 6.4, 4.4, 6.0, 3.15, tirets=True, texte="intercepte", tx=6.85, ty=3.85, fond=True)
    ax.text(6.0, 0.05, "Hypothèses : le serveur n'a pas la clé secrète et ne s'allie avec aucun hôpital.",
            ha="center", fontsize=T - 1, style="italic")
    sauver(fig, "fig3_1_modele_de_menace.png")


def fig3_2_architecture():
    fig, ax = figure(12, 7.0)
    boite(ax, 3.6, 6.0, 4.8, 0.85, "Hôpital coordinateur\ngénère la paire de clés (pk, sk)", taille=T)
    for i, y in enumerate([4.15, 2.45, 0.75]):
        boite(ax, 0.2, y, 3.7, 1.45, "", rayon=0.04)
        ax.text(0.4, y + 1.15, f"Hôpital {i + 1}", fontsize=T + 1, fontweight="bold", va="center")
        ax.text(0.4, y + 0.78, f"Données locales $D_{i + 1}$ (restent sur place)", fontsize=T - 1, va="center")
        ax.text(0.4, y + 0.45, "Modèle local, 5 époques par round", fontsize=T - 1, va="center")
        ax.text(0.4, y + 0.15, "Chiffrement, clé secrète sk", fontsize=T - 1, va="center")
        fleche(ax, 3.95, y + 1.0, 8.05, y + 1.0)
        fleche(ax, 8.05, y + 0.45, 3.95, y + 0.45, tirets=True)
        cadenas(ax, 6.0, y + 1.0)
    boite(ax, 8.1, 0.75, 3.7, 4.85, "", rayon=0.04)
    ax.text(9.95, 5.0, "Serveur d'agrégation", fontsize=T + 1, ha="center", fontweight="bold")
    ax.text(9.95, 4.55, "clé publique pk seulement", fontsize=T, ha="center", style="italic")
    ax.text(9.95, 3.3, r"$C = c_1 \oplus c_2 \oplus c_3$", fontsize=T + 4, ha="center")
    ax.text(9.95, 2.2, "additionne les messages chiffrés", fontsize=T - 1, ha="center")
    ax.text(9.95, 1.8, "ne déchiffre jamais", fontsize=T - 1, ha="center", fontweight="bold")
    fleche(ax, 4.3, 6.0, 2.2, 5.65)
    ax.text(3.3, 6.05, "sk (canal sécurisé)", fontsize=T - 1, ha="right")
    fleche(ax, 7.7, 6.0, 9.8, 5.65)
    ax.text(8.7, 6.05, "pk", fontsize=T - 1, ha="left")
    ax.text(6.0, 5.42, r"$c_k = \mathrm{Enc}_{pk}(n_k W_k)$ (trait plein)", fontsize=T, ha="center")
    ax.text(6.0, 0.4, r"$C$ renvoyé (pointillés)", fontsize=T, ha="center")
    ax.text(6.0, -0.15, r"Chaque hôpital calcule : $W = \mathrm{Dec}_{sk}(C)\,/\,n$", fontsize=T + 1, ha="center")
    sauver(fig, "fig3_2_architecture_globale.png")


def fig3_3_reseau_tabulaire():
    fig, ax = figure(12, 6.2)
    couches = [("Entrée", "d variables cliniques", 6, 1.3), ("Couche cachée 1", "32 neurones, ReLU", 8, 4.2),
               ("Couche cachée 2", "16 neurones, ReLU", 6, 7.1), ("Sortie", "1 neurone, sigmoïde", 1, 10.0)]
    positions = []
    for nom, desc, nb, x in couches:
        if nb == 1:
            ys = [3.0]
        else:
            moitie = nb // 2
            pas = 1.4 / (moitie - 1) if moitie > 1 else 0
            ys = [5.0 - i * pas for i in range(moitie)] + [1.0 + i * pas for i in range(moitie)]
            for dy in (-0.25, 0, 0.25):
                ax.add_patch(Circle((x, 3.0 + dy), 0.045, facecolor=NOIR, edgecolor="none", zorder=3))
        positions.append((x, ys))
        for y in ys:
            ax.add_patch(Circle((x, y), 0.2, facecolor=BLANC, edgecolor=NOIR, lw=1.3, zorder=3))
        ax.text(x, 5.85, nom, fontsize=T + 1, ha="center", fontweight="bold")
        ax.text(x, 5.5, desc, fontsize=T - 1, ha="center")
    for (x1, ys1), (x2, ys2) in zip(positions, positions[1:]):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + 0.2, x2 - 0.2], [y1, y2], color=GRIS, lw=0.4, zorder=1)
    ax.text(10.6, 3.0, r"$\hat{y} = P(\mathrm{malade})$", fontsize=T + 2, va="center")
    ax.text(6.0, 0.2, r"Nombre de poids : $P = (32d + 32) + (32 \times 16 + 16) + (16 + 1) = 32d + 577$",
            fontsize=T, ha="center")
    sauver(fig, "fig3_3_reseau_tabulaire.png")


def fig3_4_modele_images():
    fig, ax = figure(12, 3.6)
    blocs = [(0.2, 2.5, "Image de frottis\n224 × 224 × 3", BLANC, False),
             (3.0, 3.1, "MobileNetV2 pré-entraîné\ncouches figées\n(ni envoyées, ni chiffrées)", GRIS_CLAIR, True),
             (6.4, 2.2, "Vecteur de\n1 280 valeurs", BLANC, False),
             (8.9, 2.9, "Couche linéaire 1 280 → 1\n+ sigmoïde\n1 281 poids fédérés et chiffrés", BLANC, False)]
    for x, w, t, fond, tirets in blocs:
        boite(ax, x, 0.8, w, 2.0, t, taille=T - 1, fond=fond, tirets=tirets)
    for x1, x2 in [(2.7, 3.0), (6.1, 6.4), (8.6, 8.9)]:
        fleche(ax, x1, 1.8, x2, 1.8)
    cadenas(ax, 11.6, 2.55)
    ax.text(6.0, 0.25, "Seule la dernière couche circule entre les hôpitaux.", ha="center", fontsize=T - 1,
            style="italic")
    sauver(fig, "fig3_4_modele_images.png")


# ---------------------------------------------------------------------------
# Chapitre 4
# ---------------------------------------------------------------------------
def fig4_1_organisation_code():
    fig, ax = figure(12, 5.4)
    boite(ax, 0.2, 3.9, 2.6, 0.9, "data/\n(jeux de données)", taille=T - 1)
    boite(ax, 3.4, 3.9, 2.6, 0.9, "donnees.py\nlecture, découpage", taille=T - 1)
    boite(ax, 3.4, 2.1, 2.6, 0.9, "modele.py\nréseau, entraînement", taille=T - 1)
    boite(ax, 6.7, 2.1, 2.6, 1.8, "federe.py\nprotocole FedAvg\nmesures de temps\net de taille", taille=T - 1,
          gras=False)
    boite(ax, 3.4, 0.2, 2.6, 1.2, "chiffrement.py\nSansChiffrement, CKKS,\nBFV, Paillier", taille=T - 1)
    boite(ax, 9.9, 2.55, 1.9, 0.9, "results/\n*.csv", taille=T - 1)
    boite(ax, 9.9, 0.6, 1.9, 0.9, "figures.py", taille=T - 1)
    fleche(ax, 2.8, 4.35, 3.4, 4.35)
    fleche(ax, 6.0, 4.35, 7.4, 3.9)
    fleche(ax, 6.0, 2.55, 6.7, 2.85)
    fleche(ax, 6.0, 0.8, 7.4, 2.1, "interface commune :\nchiffrer, additionner,\ndechiffrer, taille_octets",
           tx=6.35, ty=0.45, ha="left", fond=True)
    fleche(ax, 9.3, 3.0, 9.9, 3.0)
    fleche(ax, 10.85, 2.55, 10.85, 1.5)
    sauver(fig, "fig4_1_organisation_du_code.png")


def fig4_2_deploiement_reseau():
    """Déploiement réel : un processus (ou une machine) par participant."""
    fig, ax = figure(12, 6.0)
    boite(ax, 4.0, 4.5, 4.0, 1.3, "Machine serveur : serveur.py\nclé publique seulement\n(additionne sans déchiffrer)",
          taille=T - 1)
    cadenas(ax, 7.6, 5.5)
    ax.text(8.2, 4.75, "port TCP 5050", fontsize=T - 2, style="italic", ha="left")
    hop = [(0.4, "Machine A : hopital.py\n--indice 0 (coordinateur)\ndonnées locales"),
           (4.4, "Machine B : hopital.py\n--indice 1\ndonnées locales"),
           (8.4, "Machine C : hopital.py\n--indice 2\ndonnées locales")]
    for k, (x, t) in enumerate(hop):
        boite(ax, x, 1.4, 3.2, 1.3, t, taille=T - 1)
        cx, sx = x + 1.6, 5.0 + k * 1.0
        fleche(ax, cx - 0.3, 2.7, sx - 0.15, 4.5)
        fleche(ax, sx + 0.15, 4.5, cx + 0.3, 2.7, tirets=True)
    ax.text(1.3, 3.75, "POIDS : Enc(n_k W_k)", fontsize=T - 2, ha="left")
    ax.text(10.7, 3.75, "SOMME C (chiffrée)", fontsize=T - 2, ha="right")
    ax.plot([2.0, 2.0, 10.0, 10.0], [1.4, 0.75, 0.75, 1.4], color=NOIR, lw=1.1, ls=(0, (2, 2)))
    ax.plot([6.0, 6.0], [0.75, 1.4], color=NOIR, lw=1.1, ls=(0, (2, 2)))
    for x in (6.0, 10.0):
        ax.annotate("", xy=(x, 1.4), xytext=(x, 1.1), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=NOIR))
    ax.text(6.0, 0.35, "dossier cles/ (clé secrète) copié par un canal sûr (clé USB), jamais par le serveur",
            ha="center", fontsize=T - 2, style="italic")
    sauver(fig, "fig4_2_deploiement_reseau.png")


# ---------------------------------------------------------------------------
# Chapitre 5 (graphiques à partir des résultats)
# ---------------------------------------------------------------------------
SCHEMAS = [("sans_chiffrement", "Sans chiffrement"), ("ckks", "CKKS"), ("bfv", "BFV"), ("paillier", "Paillier")]
HACHURES = {"Sans chiffrement": "", "CKKS": "///", "BFV": "...", "Paillier": "xxx"}
JEUX = {"pima": "Pima (diabète)", "heart_uci": "Heart UCI (cœur)", "ckd": "CKD (rein)"}
NB_HOPITAUX = {"pima": 3, "heart_uci": 4, "ckd": 3}
REFERENCE_CENTRALISEE = {"pima": 0.760, "heart_uci": 0.811, "ckd": 0.988}
POIDS = {"pima": 833, "heart_uci": 1409, "ckd": 1345}


def _lire(jeu, schema):
    import pandas as pd
    f = RESULTATS / f"{jeu}_{schema}.csv"
    return pd.read_csv(f) if f.exists() else None


def _axes(ax):
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    ax.tick_params(labelsize=11)


def _virgule(s):
    return s.replace(".", ",")


def _barres_par_jeu(colonne, diviser_par_hopitaux, log, xlabel, nom, xlim, fmt):
    jeux = [(j, [(n, _lire(j, c)) for c, n in SCHEMAS]) for j in JEUX]
    jeux = [(j, [(n, d) for n, d in l if d is not None]) for j, l in jeux]
    jeux = [(j, l) for j, l in jeux if l]
    if not jeux:
        return
    fig, axes = plt.subplots(1, len(jeux), figsize=(4.3 * len(jeux), 3.4))
    axes = axes if len(jeux) > 1 else [axes]
    for ax, (jeu, lignes) in zip(axes, jeux):
        noms = [n for n, _ in lignes]
        vals = [d[colonne].mean() / (NB_HOPITAUX[jeu] if diviser_par_hopitaux else 1) for _, d in lignes]
        y = list(range(len(noms)))[::-1]
        for yi, n, v in zip(y, noms, vals):
            ax.barh(yi, v, height=0.62, facecolor="white", edgecolor="black", hatch=HACHURES[n], lw=1.0)
        if log:
            ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_yticks(y)
        ax.set_yticklabels(noms if ax is axes[0] else [], fontsize=11)
        for yi, v in zip(y, vals):
            ax.text(v * 1.4 if log else v + xlim[1] * 0.02, yi, _virgule(fmt(v)), va="center", fontsize=10.5)
        ax.set_title(JEUX[jeu], fontsize=13)
        ax.grid(axis="x", color="#CCCCCC", lw=0.6)
        ax.set_axisbelow(True)
        _axes(ax)
    fig.supxlabel(xlabel, fontsize=12)
    fig.tight_layout()
    sauver(fig, nom)


def fig5_1_convergence():
    jeux = [(j, _lire(j, "sans_chiffrement")) for j in JEUX]
    jeux = [(j, d) for j, d in jeux if d is not None]
    if not jeux:
        return
    fig, axes = plt.subplots(1, len(jeux), figsize=(4 * len(jeux), 3.5))
    axes = axes if len(jeux) > 1 else [axes]
    for ax, (jeu, d) in zip(axes, jeux):
        ax.plot(d["round"], d["accuracy"], color="black", lw=1.6, marker="o", ms=4.5,
                label="FL (avec ou sans chiffrement)")
        ref = REFERENCE_CENTRALISEE[jeu]
        ax.axhline(ref, color="black", lw=1.2, ls=(0, (5, 4)), label="Référence centralisée")
        ax.set_title(JEUX[jeu], fontsize=13)
        ax.set_xlabel("Round", fontsize=11)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(min(d["accuracy"].min(), ref) - 0.04, min(1.0, max(d["accuracy"].max(), ref) + 0.03))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color="#CCCCCC", lw=0.6)
        _axes(ax)
    axes[0].set_ylabel("Accuracy (jeu de test)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False, loc="lower right")
    fig.tight_layout()
    sauver(fig, "fig5_1_convergence.png")


def fig5_2_temps():
    """Temps de chiffrement par hôpital et par round : moyenne ± écart-type sur 5 répétitions
    (results/repetitions/resume.csv). Sans ce fichier, on revient à la répétition unique."""
    import pandas as pd
    f = RESULTATS / "repetitions" / "resume.csv"
    if not f.exists():
        _barres_par_jeu("t_chiffrement", True, True,
                        "Temps moyen de chiffrement par hôpital et par round (s, échelle logarithmique)",
                        "fig5_2_temps_chiffrement.png", (1e-4, 5e3),
                        lambda t: f"{t:.3f} s" if t < 1 else f"{t:.0f} s")
        return
    r = pd.read_csv(f)
    codes = {"sans_chiffrement": "sans", "ckks": "ckks", "bfv": "bfv", "paillier": "paillier"}
    fig, axes = plt.subplots(1, len(JEUX), figsize=(4.3 * len(JEUX), 3.4))
    for ax, jeu in zip(axes, JEUX):
        y = list(range(len(SCHEMAS)))[::-1]
        for yi, (c, n) in zip(y, SCHEMAS):
            ligne = r[(r["jeu"] == jeu) & (r["schema"] == codes[c])]
            if ligne.empty:
                continue
            m = ligne["t_chiffrement_hopital_moyenne"].iloc[0]
            e = ligne["t_chiffrement_hopital_ecart_type"].iloc[0]
            ax.barh(yi, m, height=0.62, facecolor="white", edgecolor="black", hatch=HACHURES[n], lw=1.0)
            ax.errorbar(m, yi, xerr=[[min(e, m * 0.9)], [e]], fmt="none", ecolor="black", capsize=4, lw=1.0)
            if m < 0.01:
                texte = f"{m:.4f} s"
            elif m < 1:
                texte = f"{m:.3f} s"
            else:
                texte = f"{m:.0f} ± {e:.0f} s" if e >= 1 else f"{m:.1f} ± {e:.1f} s"
            ax.text((m + e) * 1.5, yi, _virgule(texte), va="center", fontsize=10.5)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 2e4)
        ax.set_yticks(y)
        ax.set_yticklabels([n for _, n in SCHEMAS] if ax is axes[0] else [], fontsize=11)
        ax.set_title(JEUX[jeu], fontsize=13)
        ax.grid(axis="x", color="#CCCCCC", lw=0.6)
        ax.set_axisbelow(True)
        _axes(ax)
    fig.supxlabel("Temps de chiffrement par hôpital et par round (s, échelle logarithmique ; "
                  "moyenne ± écart-type, 5 répétitions)", fontsize=11)
    fig.tight_layout()
    sauver(fig, "fig5_2_temps_chiffrement.png")


def fig5_5_round_reseau():
    """Durée d'un round complet dans le système en réseau (results/reseau/resume.csv)."""
    import pandas as pd
    f = RESULTATS / "reseau" / "resume.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    codes = {"sans_chiffrement": "sans", "ckks": "ckks", "bfv": "bfv", "paillier": "paillier"}
    fig, axes = plt.subplots(1, len(JEUX), figsize=(4.3 * len(JEUX), 3.4))
    for ax, jeu in zip(axes, JEUX):
        y = list(range(len(SCHEMAS)))[::-1]
        for yi, (c, n) in zip(y, SCHEMAS):
            ligne = r[(r["jeu"] == jeu) & (r["schema"] == codes[c])]
            if ligne.empty:
                continue
            v = ligne["t_round"].iloc[0]
            ax.barh(yi, v, height=0.62, facecolor="white", edgecolor="black", hatch=HACHURES[n], lw=1.0)
            ax.text(v * 1.3, yi, _virgule(f"{v:.2f} s" if v < 10 else f"{v:.0f} s"), va="center", fontsize=10.5)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 2000)
        ax.set_yticks(y)
        ax.set_yticklabels([n for _, n in SCHEMAS] if ax is axes[0] else [], fontsize=11)
        ax.set_title(JEUX[jeu], fontsize=13)
        ax.grid(axis="x", color="#CCCCCC", lw=0.6)
        ax.set_axisbelow(True)
        _axes(ax)
    fig.supxlabel("Durée d'un round complet en réseau, vue par l'hôpital coordinateur (s, échelle logarithmique)",
                  fontsize=11)
    fig.tight_layout()
    sauver(fig, "fig5_5_round_reseau.png")


def fig5_3_taille():
    # Version en Ko (plus lisible)
    jeux = [(j, [(n, _lire(j, c)) for c, n in SCHEMAS]) for j in JEUX]
    jeux = [(j, [(n, d) for n, d in l if d is not None]) for j, l in jeux]
    jeux = [(j, l) for j, l in jeux if l]
    fig, axes = plt.subplots(1, len(jeux), figsize=(4.3 * len(jeux), 3.4))
    for ax, (jeu, lignes) in zip(axes, jeux):
        noms = [n for n, _ in lignes]
        vals = [d["octets_par_message"].mean() / 1024 for _, d in lignes]
        y = list(range(len(noms)))[::-1]
        for yi, n, v in zip(y, noms, vals):
            ax.barh(yi, v, height=0.62, facecolor="white", edgecolor="black", hatch=HACHURES[n], lw=1.0)
        ax.set_xlim(0, 1500)
        ax.set_yticks(y)
        ax.set_yticklabels(noms if ax is axes[0] else [], fontsize=11)
        for yi, v in zip(y, vals):
            ax.text(v + 30, yi, f"{v:.0f} Ko", va="center", fontsize=10.5)
        ax.set_title(JEUX[jeu], fontsize=13)
        ax.grid(axis="x", color="#CCCCCC", lw=0.6)
        ax.set_axisbelow(True)
        _axes(ax)
    fig.supxlabel("Taille d'un message envoyé par un hôpital (Ko)", fontsize=12)
    fig.tight_layout()
    sauver(fig, "fig5_3_taille_messages.png")


def fig5_4_taille_selon_poids():
    P = list(range(0, 2001, 10))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(P, [768 * p / 1024 for p in P], color="black", lw=1.8)
    ax.plot(P, [642.5] * len(P), color="black", lw=1.4, ls=(0, (7, 4)))
    ax.plot(P, [422.3] * len(P), color="black", lw=1.4, ls=(0, (1.5, 2.5)))
    ax.text(1990, 1500, "Paillier : 0,75 Ko par poids", fontsize=12, ha="right")
    ax.text(1990, 642.5 + 35, "CKKS : 642 Ko", fontsize=12, ha="right")
    ax.text(1990, 422.3 - 85, "BFV : 422 Ko", fontsize=12, ha="right")
    for jeu, p in POIDS.items():
        ax.plot([p], [768 * p / 1024], "o", color="black", ms=7, mfc="white", mew=1.5, zorder=4)
        dec = {"pima": (12, -22), "ckd": (-10, 12), "heart_uci": (10, -16)}[jeu]
        ax.annotate(f"{JEUX[jeu].split(' (')[0]} (P = {p})", (p, 768 * p / 1024), xytext=dec,
                    textcoords="offset points", ha="right" if jeu == "ckd" else "left", fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    for seuil in (422.3 * 1024 / 768, 642.5 * 1024 / 768):
        ax.axvline(seuil, color="#777777", lw=0.9, ls=":")
        ax.text(seuil + 12, 60, f"P = {seuil:.0f}", fontsize=11, color="#444444")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1600)
    ax.set_xlabel("Nombre de poids du modèle (P)", fontsize=12)
    ax.set_ylabel("Taille d'un message (Ko)", fontsize=12)
    ax.grid(axis="y", color="#CCCCCC", lw=0.6)
    _axes(ax)
    sauver(fig, "fig5_4_taille_selon_poids.png")


def fig5_6_attaque():
    """Attaque par inversion de gradients : ré-identification et exemple d'un patient (results/attaque/)."""
    import pandas as pd
    f_resume, f_exemple = RESULTATS / "attaque" / "resume.csv", RESULTATS / "attaque" / "exemple.csv"
    if not (f_resume.exists() and f_exemple.exists()):
        return
    r = pd.read_csv(f_resume)
    r = r[r["jeu"] == "pima"]
    ex = pd.read_csv(f_exemple)
    ex = ex[ex["jeu"] == "pima"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})

    clair = r[r["configuration"] == "sans"].sort_values("taille_lot")
    hasard = r[r["configuration"] == "hasard"].sort_values("taille_lot")
    a1.plot(clair["taille_lot"], 100 * clair["reidentification"], color="black", lw=1.8, marker="o",
            ms=6, label="Sans chiffrement")
    a1.plot(hasard["taille_lot"], 100 * hasard["reidentification"], color="black", lw=1.2, ls=(0, (5, 4)),
            label="Hasard (référence)")
    marqueurs = {"ckks": ("s", "CKKS"), "bfv": ("^", "BFV"), "paillier": ("D", "Paillier")}
    for cfg, (m, nom) in marqueurs.items():
        d = r[r["configuration"] == cfg]
        if not d.empty:
            a1.plot(d["taille_lot"], 100 * d["reidentification"], ls="none", marker=m, ms=8, mfc="white",
                    mec="black", mew=1.4, label=f"{nom} (clé de l'attaquant)")
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8, 16])
    a1.set_xticklabels(["1", "2", "4", "8", "16"])
    a1.set_ylim(-3, 105)
    a1.set_xlabel("Nombre de patients dans le lot (B)", fontsize=11)
    a1.set_ylabel("Patients ré-identifiés (%)", fontsize=11)
    a1.set_title("(a) Taux de ré-identification (Pima)", fontsize=12)
    a1.legend(fontsize=9, frameon=False, loc="upper right")
    a1.grid(axis="y", color="#CCCCCC", lw=0.6)
    _axes(a1)

    noms = {"Pregnancies": "Grossesses", "Glucose": "Glucose", "BloodPressure": "Tension",
            "SkinThickness": "Pli cutané", "Insulin": "Insuline", "BMI": "IMC",
            "DiabetesPedigreeFunction": "Hérédité", "Age": "Âge"}
    x = np.arange(len(ex))
    l = 0.27
    a2.bar(x - l, ex["vrai"], l, facecolor="white", edgecolor="black", lw=1.0, label="Vrai patient")
    a2.bar(x, ex["reconstruit_sans_chiffrement"], l, facecolor="#BBBBBB", edgecolor="black", lw=1.0,
           label="Reconstruit, sans chiffrement")
    a2.bar(x + l, ex["reconstruit_avec_ckks"], l, facecolor="white", edgecolor="black", hatch="xxx", lw=1.0,
           label="Reconstruit, avec CKKS")
    if "vrai_unite" in ex:
        for xi, (v, vu) in enumerate(zip(ex["vrai"], ex["vrai_unite"])):
            txt = f"{vu:.0f}" if abs(vu) >= 10 else _virgule(f"{vu:.1f}" if abs(vu) >= 1 else f"{vu:.2f}")
            a2.text(xi - l, (v if v > 0 else 0) + 0.12, txt, ha="center", fontsize=9)
    a2.axhline(0, color="black", lw=0.8)
    a2.set_xticks(x)
    a2.set_xticklabels([noms.get(v, v) for v in ex["variable"]], fontsize=10, rotation=20)
    a2.set_ylabel("Valeur standardisée", fontsize=11)
    a2.set_title("(b) Un patient (B = 1) : valeurs réelles au-dessus des barres", fontsize=12)
    a2.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    a2.grid(axis="y", color="#CCCCCC", lw=0.6)
    a2.set_axisbelow(True)
    _axes(a2)
    fig.tight_layout()
    sauver(fig, "fig5_6_attaque_inversion.png")


if __name__ == "__main__":
    for f in [fig1_1_principe_fl, fig1_2_inversion_gradients, fig1_3_techniques_protection,
              fig1_4_principe_chiffrement_homomorphe, fig2_1_cas_utilisation, fig2_2_sequence,
              fig3_1_modele_menace, fig3_2_architecture, fig3_3_reseau_tabulaire, fig3_4_modele_images,
              fig4_1_organisation_code, fig4_2_deploiement_reseau, fig5_1_convergence, fig5_2_temps, fig5_3_taille,
              fig5_4_taille_selon_poids, fig5_5_round_reseau,
              fig5_6_attaque]:
        f()
    print("Figures créées dans", OUT)