"""Figures du mémoire (architecture, séquence, modèles, cas d'utilisation, formules).

Utilisation : python -m src.figures
Les images sont créées dans le dossier figures/ à la racine du projet.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = ["Calibri", "Carlito", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "cm"

VERT_FONCE = "#0B4D2C"
VERT = "#00853F"
VERT_CLAIR = "#EDF5EF"
JAUNE = "#F2C200"
ROUGE = "#C8102E"
GRIS = "#7D8C83"
TEXTE = "#1A2B22"
SAUGE = "#4F7A5E"


def boite(ax, x, y, w, h, fond, bord=None, rayon=0.02, lw=1.2, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rayon}",
                                facecolor=fond, edgecolor=bord or fond, linewidth=lw, zorder=z))


def fleche(ax, x1, y1, x2, y2, couleur, texte=None, tx=None, ty=None, taille=12, lw=2.2, style="-|>", ha="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=couleur, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=18))
    if texte:
        ax.text(tx if tx is not None else (x1 + x2) / 2, ty if ty is not None else (y1 + y2) / 2, texte,
                fontsize=taille, color=couleur, ha=ha, va="center", fontweight="bold")


def nouvelle_figure(l=12, h=6.75):
    fig, ax = plt.subplots(figsize=(l, h))
    ax.set_xlim(0, l)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def sauver(fig, nom):
    fig.savefig(OUT / nom, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 : architecture globale du système proposé
# ---------------------------------------------------------------------------
def architecture_globale():
    fig, ax = nouvelle_figure(12, 7.0)
    boite(ax, 3.6, 6.0, 4.8, 0.85, VERT_FONCE)
    ax.text(6.0, 6.55, "Hôpital coordinateur", fontsize=15, color="white", ha="center", va="center", fontweight="bold")
    ax.text(6.0, 6.22, "génère la paire de clés (pk, sk)", fontsize=12.5, color=JAUNE, ha="center", va="center")
    for i, y in enumerate([4.15, 2.45, 0.75]):
        boite(ax, 0.2, y, 3.7, 1.45, VERT_CLAIR, VERT, lw=1.5)
        ax.text(0.4, y + 1.18, f"Hôpital {i + 1}", fontsize=14.5, color=VERT_FONCE, fontweight="bold", va="center")
        ax.text(0.4, y + 0.85, f"Données locales D{chr(0x2081 + i)} (ne sortent jamais)", fontsize=11.5, color=TEXTE, va="center")
        ax.text(0.4, y + 0.55, "Modèle local · 5 époques par round", fontsize=11.5, color=TEXTE, va="center")
        ax.text(0.4, y + 0.25, "Chiffrement · clé secrète sk", fontsize=11.5, color=SAUGE, va="center")
        fleche(ax, 3.95, y + 1.0, 8.05, y + 1.0, VERT, lw=2.0)
        fleche(ax, 8.05, y + 0.45, 3.95, y + 0.45, "#C9A100", lw=2.0)
    boite(ax, 8.1, 0.75, 3.7, 4.85, VERT_FONCE)
    ax.text(9.95, 5.0, "Serveur d'agrégation", fontsize=15, color="white", ha="center", fontweight="bold")
    ax.text(9.95, 4.55, "contexte public pk seulement", fontsize=12.5, color=JAUNE, ha="center")
    ax.text(9.95, 3.3, r"$C = c_1 \oplus c_2 \oplus c_3$", fontsize=18, color="white", ha="center")
    ax.text(9.95, 2.2, "additionne les messages chiffrés", fontsize=12, color="white", ha="center")
    ax.text(9.95, 1.8, "ne déchiffre jamais", fontsize=12, color="white", ha="center", fontweight="bold")
    fleche(ax, 4.3, 6.0, 2.2, 5.65, SAUGE, lw=1.6)
    ax.text(3.3, 6.05, "sk (canal sécurisé)", fontsize=11.5, color=SAUGE, ha="right", fontweight="bold")
    fleche(ax, 7.7, 6.0, 9.8, 5.65, SAUGE, lw=1.6)
    ax.text(8.7, 6.05, "pk (clé publique)", fontsize=11.5, color=SAUGE, ha="left", fontweight="bold")
    ax.text(6.0, 5.4, r"$c_k = \mathrm{Enc}_{pk}(n_k \cdot W_k)$", fontsize=15, color=VERT, ha="center")
    ax.text(6.0, 0.88, r"$C = \mathrm{Enc}_{pk}(\Sigma_k\, n_k W_k)$ renvoyé", fontsize=14, color="#8A6D00", ha="center", va="top")
    ax.text(6.0, 0.05, r"Chaque hôpital calcule : $W_{global} = \mathrm{Dec}_{sk}(C)\,/\,n$", fontsize=14.5,
            color=VERT_FONCE, ha="center", fontweight="bold")
    sauver(fig, "fig1_architecture_globale.png")


# ---------------------------------------------------------------------------
# Figure 2 : diagramme de séquence d'un round
# ---------------------------------------------------------------------------
def sequence():
    fig, ax = nouvelle_figure(12, 7.4)
    acteurs = {"Hôpital coordinateur": 1.6, "Hôpital k (k = 1 … K)": 6.0, "Serveur d'agrégation": 10.25}
    for nom, x in acteurs.items():
        boite(ax, x - 1.45, 6.75, 2.9, 0.55, VERT_FONCE)
        ax.text(x, 7.02, nom, fontsize=13, color="white", ha="center", va="center", fontweight="bold")
        ax.plot([x, x], [0.2, 6.75], color=GRIS, lw=1.2, ls=(0, (5, 4)), zorder=1)
    xc, xh, xs = acteurs.values()

    def message(x1, x2, y, texte, couleur=TEXTE):
        fleche(ax, x1, y, x2, y, couleur, lw=1.7)
        ax.text((x1 + x2) / 2, y + 0.16, texte, fontsize=12, color=couleur, ha="center", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))

    def action(x, y, texte, w=3.1):
        boite(ax, x - w / 2, y - 0.2, w, 0.4, VERT_CLAIR, VERT, rayon=0.05, lw=1, z=3)
        ax.text(x, y, texte, fontsize=11.5, color=TEXTE, ha="center", va="center", zorder=4)

    action(xc, 6.35, "génère (pk, sk)", w=2.2)
    message(xc, xh, 5.85, "sk  (canal sécurisé)", SAUGE)
    message(xc, xs, 5.35, "pk  (contexte public)", SAUGE)
    # boucle
    boite(ax, 0.1, 0.3, 11.85, 4.65, "none", JAUNE, rayon=0.03, lw=1.6, z=1)
    boite(ax, 0.1, 4.6, 2.65, 0.35, JAUNE, rayon=0.02, z=3)
    ax.text(1.42, 4.775, "boucle : round t = 1 … 10", fontsize=11.5, color=TEXTE, ha="center", va="center", fontweight="bold", zorder=4)
    action(xh, 4.25, "1. entraînement local (5 époques)")
    action(xh, 3.7, r"2. $c_k = \mathrm{Enc}_{pk}(n_k \cdot W_k)$")
    message(xh, xs, 3.2, r"3. $c_k$, $n_k$", VERT)
    action(xs, 2.65, r"4. $C = c_1 \oplus \cdots \oplus c_K$")
    message(xs, xh, 2.1, r"5. $C$, $n$", "#8A6D00")
    action(xh, 1.5, r"6. $W_{global} = \mathrm{Dec}_{sk}(C)/n$")
    action(xh, 0.85, "7. évaluation sur le jeu de test")
    sauver(fig, "fig2_sequence_round.png")


# ---------------------------------------------------------------------------
# Figure 3 : architecture du réseau de neurones (données tabulaires)
# ---------------------------------------------------------------------------
def reseau_tabulaire():
    fig, ax = nouvelle_figure(12, 6.2)
    couches = [("Entrée", "d variables cliniques", 6, 1.3),
               ("Couche dense 1", "32 neurones + ReLU", 8, 4.2),
               ("Couche dense 2", "16 neurones + ReLU", 6, 7.1),
               ("Sortie", "1 neurone + sigmoïde", 1, 10.0)]
    positions = []
    for nom, desc, nb, x in couches:
        if nb == 1:
            ys = [3.0]
        else:
            moitie = nb // 2
            pas = 1.4 / (moitie - 1) if moitie > 1 else 0
            ys = [5.0 - i * pas for i in range(moitie)] + [1.0 + i * pas for i in range(moitie)]
            for dy in (-0.25, 0, 0.25):
                ax.add_patch(Circle((x, 3.0 + dy), 0.045, facecolor=GRIS, edgecolor="none", zorder=3))
        positions.append((x, ys))
        for y in ys:
            ax.add_patch(Circle((x, y), 0.2, facecolor=VERT if nom != "Sortie" else VERT_FONCE, edgecolor="white", lw=1.5, zorder=3))
        ax.text(x, 5.85, nom, fontsize=14.5, color=VERT_FONCE, ha="center", fontweight="bold")
        ax.text(x, 5.5, desc, fontsize=12, color=TEXTE, ha="center")
    for (x1, ys1), (x2, ys2) in zip(positions, positions[1:]):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1 + 0.2, x2 - 0.2], [y1, y2], color="#B9CEC0", lw=0.6, zorder=1)
    ax.text(10.65, 3.0, r"$\hat{y} = P(\mathrm{malade})$", fontsize=15, color=VERT_FONCE, va="center")
    ax.text(6.0, 0.2, r"Nombre de poids : $P = (32d + 32) + (32 \times 16 + 16) + (16 + 1) = 32d + 577$",
            fontsize=14, color=TEXTE, ha="center")
    sauver(fig, "fig3_reseau_tabulaire.png")


# ---------------------------------------------------------------------------
# Figure 4 : architecture du modèle pour les images de drépanocytose
# ---------------------------------------------------------------------------
def modele_images():
    fig, ax = nouvelle_figure(12, 4.6)
    blocs = [
        (0.2, 2.5, "Image de frottis", "redimensionnée\n224 × 224 × 3", VERT_CLAIR, TEXTE),
        (3.0, 3.1, "MobileNetV2 pré-entraîné", "(ImageNet) · couches figées\nnon envoyées, non chiffrées", GRIS, "white"),
        (6.4, 2.2, "Vecteur de\ncaractéristiques", "1 280 valeurs", VERT_CLAIR, TEXTE),
        (8.9, 2.9, "Couche linéaire 1 280 → 1", "+ sigmoïde · 1 281 poids\nentraînés en fédéré et chiffrés", VERT, "white"),
    ]
    for x, w, titre, desc, fond, coul in blocs:
        boite(ax, x, 1.3, w, 2.2, fond, VERT_FONCE if fond == VERT_CLAIR else fond, lw=1.4)
        ax.text(x + w / 2, 2.85, titre, fontsize=13.5, color=coul, ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, 1.95, desc, fontsize=11.5, color=coul, ha="center", va="center")
    for x1, x2 in [(2.7, 3.0), (6.1, 6.4), (8.6, 8.9)]:
        fleche(ax, x1, 2.4, x2, 2.4, VERT_FONCE, lw=2)
    ax.text(6.0, 0.55, "Seule la dernière couche circule entre les hôpitaux : 1 281 poids à chiffrer par round.",
            fontsize=13.5, color=VERT_FONCE, ha="center", fontweight="bold")
    ax.text(6.0, 4.15, "Modèle proposé pour la drépanocytose (implémentation à venir)", fontsize=14, color=SAUGE,
            ha="center", style="italic")
    sauver(fig, "fig4_modele_images.png")


# ---------------------------------------------------------------------------
# Figure 5 : diagramme de cas d'utilisation (UML)
# ---------------------------------------------------------------------------
def acteur(ax, x, y, nom):
    ax.add_patch(Circle((x, y + 0.75), 0.17, facecolor="white", edgecolor=VERT_FONCE, lw=1.8))
    ax.plot([x, x], [y + 0.58, y + 0.15], color=VERT_FONCE, lw=1.8)
    ax.plot([x - 0.3, x + 0.3], [y + 0.45, y + 0.45], color=VERT_FONCE, lw=1.8)
    ax.plot([x, x - 0.25], [y + 0.15, y - 0.2], color=VERT_FONCE, lw=1.8)
    ax.plot([x, x + 0.25], [y + 0.15, y - 0.2], color=VERT_FONCE, lw=1.8)
    ax.text(x, y - 0.45, nom, fontsize=12, color=VERT_FONCE, ha="center", va="center", fontweight="bold")


def cas_utilisation():
    from matplotlib.patches import Ellipse
    fig, ax = nouvelle_figure(12, 7.2)
    boite(ax, 2.55, 0.25, 6.9, 6.7, "none", VERT_FONCE, rayon=0.03, lw=1.6)
    ax.text(6.0, 6.65, "Système d'apprentissage fédéré sécurisé", fontsize=14, color=VERT_FONCE, ha="center", fontweight="bold")
    cas = [
        (6.0, 6.0, "Générer les clés (pk, sk)"),
        (6.0, 5.2, "Entraîner le modèle localement"),
        (6.0, 4.4, "Chiffrer les paramètres"),
        (6.0, 3.6, "Agréger les paramètres chiffrés"),
        (6.0, 2.8, "Déchiffrer le modèle global"),
        (6.0, 2.0, "Évaluer le modèle"),
        (6.0, 1.2, "Comparer CKKS, BFV et Paillier"),
        (6.0, 0.55, "Simuler une attaque par inversion"),
    ]
    for x, y, t in cas:
        ax.add_patch(Ellipse((x, y), 5.4, 0.55, facecolor=VERT_CLAIR, edgecolor=VERT, lw=1.3))
        ax.text(x, y, t, fontsize=12, color=TEXTE, ha="center", va="center")
    acteur(ax, 1.0, 5.4, "Hôpital\ncoordinateur")
    acteur(ax, 1.0, 3.3, "Hôpital\n(client)")
    acteur(ax, 11.0, 3.3, "Serveur\nd'agrégation")
    acteur(ax, 11.0, 0.9, "Expérimentatrice")
    lien = dict(color=GRIS, lw=1.1)
    for y in [6.0]:
        ax.plot([1.3, 3.3], [5.75, y], **lien)
    for y in [5.2, 4.4, 2.8, 2.0]:
        ax.plot([1.3, 3.3], [3.65, y], **lien)
    for y in [3.6]:
        ax.plot([10.7, 8.7], [3.65, y], **lien)
    for y in [1.2, 0.55]:
        ax.plot([10.7, 8.7], [1.25, y], **lien)
    sauver(fig, "fig5_cas_utilisation.png")


# ---------------------------------------------------------------------------
# Formules
# ---------------------------------------------------------------------------
FORMULES = {
    "eq1_fedavg": r"$W_{global}^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{n}\, W_k^{(t)} \qquad n = \sum_{k=1}^{K} n_k$",
    "eq2_chiffrement": r"$c_k = \mathrm{Enc}_{pk}\left(n_k\, W_k\right)$",
    "eq3_agregation": r"$C = c_1 \oplus c_2 \oplus \cdots \oplus c_K = \mathrm{Enc}_{pk}\left(\sum_{k=1}^{K} n_k\, W_k\right)$",
    "eq4_dechiffrement": r"$W_{global} = \frac{1}{n}\, \mathrm{Dec}_{sk}(C)$",
    "eq5_paillier_enc": r"$\mathrm{Enc}(m) = g^{m}\, r^{N} \ \mathrm{mod}\  N^2$",
    "eq5_paillier_add": r"$\mathrm{Enc}(m_1)\cdot \mathrm{Enc}(m_2) = \mathrm{Enc}(m_1+m_2) \ \mathrm{mod}\  N^2$",
    "eq6_encodage": r"$m = \mathrm{arrondi}\left(w \times 10^{7}\right) \qquad w \approx m \,/\, 10^{7}$",
    "eq7_modele": r"$\hat{y} = \sigma\left(W_3\, \mathrm{ReLU}\left(W_2\, \mathrm{ReLU}(W_1 x + b_1) + b_2\right) + b_3\right)$",
    "eq8_perte": r"$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$",
    "eq9_taille_paillier": r"$\mathrm{Taille}_{Paillier} = P \times \frac{2 \times 3072}{8}\ \mathrm{octets} = 768\,P\ \mathrm{octets}$",
    "eq10_f1": r"$F_1 = \frac{2 \times \mathrm{Pr\acute{e}cision} \times \mathrm{Rappel}}{\mathrm{Pr\acute{e}cision} + \mathrm{Rappel}}$",
}


def formules():
    for nom, f in FORMULES.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f, fontsize=26, color=VERT_FONCE)
        fig.savefig(OUT / f"{nom}.png", dpi=200, bbox_inches="tight", pad_inches=0.08, transparent=True)
        plt.close(fig)


if __name__ == "__main__":
    architecture_globale()
    sequence()
    reseau_tabulaire()
    modele_images()
    cas_utilisation()
    formules()
    print("Figures créées dans", OUT)