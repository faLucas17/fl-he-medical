"""Tableau de bord web : suivre le système d'apprentissage fédéré chiffré en direct.

Le tableau de bord est une petite page web (dossier tableau/). Le serveur et les
hôpitaux lui envoient des événements (src/evenements.py) ; la page les affiche en
direct : hôpitaux connectés, messages chiffrés qui circulent, ce que voit le
serveur, accuracy à chaque round. Elle s'adapte à l'ordinateur, à la tablette et
au téléphone.

Utilisation :
    python -m src.tableau_de_bord                 # puis ouvrir http://127.0.0.1:8080
    python -m src.tableau_de_bord --port 8080 --sans-lancement

Depuis la page, le bouton « Lancer l'entraînement » démarre le serveur et les
hôpitaux sur cet ordinateur (comme src.lancer_reseau). Avec --sans-lancement
(utilisé par Docker), le bouton est masqué : ce sont les conteneurs qui lancent
le serveur et les hôpitaux.

Pour voir la page depuis un téléphone ou une tablette du même Wi-Fi, ouvrir
http://<adresse IP de cet ordinateur>:8080.

Seule la bibliothèque standard de Python est utilisée.
"""
import argparse
import json
import queue
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RACINE = Path(__file__).resolve().parent.parent
PAGE = RACINE / "tableau" / "index.html"
JEUX = ["pima", "heart_uci", "ckd"]
SCHEMAS = ["sans", "ckks", "bfv", "paillier"]


class Etat:
    """Historique des événements de l'exécution en cours, et abonnés de la page."""

    def __init__(self):
        self.verrou = threading.Lock()
        self.historique = []
        self.abonnes = []
        self.processus = None

    def publier(self, evenement):
        with self.verrou:
            if evenement.get("type") == "serveur_demarre":
                self.historique = []  # nouvelle exécution : on repart de zéro
            self.historique.append(evenement)
            for file in list(self.abonnes):
                file.put(evenement)

    def abonner(self):
        file = queue.Queue()
        with self.verrou:
            for evenement in self.historique:
                file.put(evenement)
            self.abonnes.append(file)
        return file

    def desabonner(self, file):
        with self.verrou:
            if file in self.abonnes:
                self.abonnes.remove(file)

    def en_cours(self):
        return self.processus is not None and self.processus.poll() is None


def adresses_locales():
    """Adresses IP de cet ordinateur sur le réseau local (pour le téléphone)."""
    adresses = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        adresses.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(a for a in adresses if not a.startswith("127."))


def creer_gestionnaire(etat, port, lancement):
    class Gestionnaire(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # pas de journal à chaque requête
            pass

        def _repondre(self, code, contenu, type_="application/json; charset=utf-8"):
            donnees = contenu if isinstance(contenu, bytes) else json.dumps(contenu).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", type_)
            self.send_header("Content-Length", str(len(donnees)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(donnees)

        def do_GET(self):
            chemin = urlparse(self.path).path
            if chemin in ("/", "/index.html"):
                self._repondre(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif chemin == "/config":
                self._repondre(200, {"lancement": lancement, "en_cours": etat.en_cours(),
                                     "adresses": [f"http://{a}:{port}" for a in adresses_locales()],
                                     "jeux": JEUX, "schemas": SCHEMAS})
            elif chemin == "/flux":
                self._flux()
            else:
                self._repondre(404, {"erreur": "page inconnue"})

        def do_POST(self):
            url = urlparse(self.path)
            longueur = int(self.headers.get("Content-Length") or 0)
            corps = self.rfile.read(longueur) if longueur else b""
            if url.path == "/evenement":
                try:
                    etat.publier(json.loads(corps.decode("utf-8")))
                    self._repondre(200, {"ok": True})
                except ValueError:
                    self._repondre(400, {"erreur": "JSON invalide"})
            elif url.path == "/lancer":
                self._lancer(parse_qs(url.query))
            else:
                self._repondre(404, {"erreur": "page inconnue"})

        def _lancer(self, q):
            if not lancement:
                return self._repondre(403, {"erreur": "Le lancement depuis la page est désactivé (mode Docker)."})
            if etat.en_cours():
                return self._repondre(409, {"erreur": "Un entraînement est déjà en cours."})
            jeu = q.get("jeu", ["pima"])[0]
            schema = q.get("schema", ["bfv"])[0]
            pause = q.get("pause", ["1"])[0]
            if jeu not in JEUX or schema not in SCHEMAS:
                return self._repondre(400, {"erreur": "Jeu ou schéma inconnu."})
            try:
                pause = str(max(0.0, min(float(pause), 10.0)))
            except ValueError:
                pause = "1"
            commande = [sys.executable, "-m", "src.lancer_reseau", "--jeu", jeu, "--schema", schema,
                        "--tableau", f"http://127.0.0.1:{port}", "--pause", pause]
            etat.processus = subprocess.Popen(commande, cwd=RACINE)
            self._repondre(200, {"ok": True, "jeu": jeu, "schema": schema})

        def _flux(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            file = etat.abonner()
            try:
                while True:
                    try:
                        evenement = file.get(timeout=15)
                        message = f"data: {json.dumps(evenement)}\n\n"
                    except queue.Empty:
                        message = ": ping\n\n"  # garde la connexion ouverte
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                etat.desabonner(file)

    return Gestionnaire


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sans-lancement", action="store_true", help="masquer le bouton de lancement (Docker)")
    args = parser.parse_args()
    etat = Etat()
    serveur = ThreadingHTTPServer(("0.0.0.0", args.port), creer_gestionnaire(etat, args.port, not args.sans_lancement))
    serveur.daemon_threads = True
    print(f"Tableau de bord : http://127.0.0.1:{args.port}", flush=True)
    for adresse in adresses_locales():
        print(f"Depuis un téléphone ou une tablette du même Wi-Fi : http://{adresse}:{args.port}", flush=True)
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        print("\nTableau de bord arrêté.")


if __name__ == "__main__":
    main()
