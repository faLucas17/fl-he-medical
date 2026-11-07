"""Envoi d'événements vers le tableau de bord (src/tableau_de_bord.py).

Le serveur et les hôpitaux peuvent signaler ce qu'ils font (connexion, round,
message reçu...) à un tableau de bord web. L'envoi se fait en arrière-plan :
si le tableau de bord n'est pas lancé, le système fonctionne exactement pareil.

Aucune donnée de patient ni aucun poids en clair n'est envoyé : seulement des
mesures (temps, tailles, accuracy) et, côté serveur, un aperçu des octets qu'il
reçoit, c'est-à-dire ce qu'il voit déjà.
"""
import json
import queue
import threading
import time
import urllib.request


class Diffuseur:
    def __init__(self, url=None, source="?"):
        self.url = url.rstrip("/") + "/evenement" if url else None
        self.source = source
        self._file = queue.Queue()
        # Connexion directe, sans passer par un éventuel proxy de l'ordinateur.
        self._ouvreur = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if self.url:
            threading.Thread(target=self._boucle, daemon=True).start()

    def envoyer(self, type_, **donnees):
        if self.url:
            self._file.put({"type": type_, "source": self.source, "heure": time.time(), **donnees})

    def vider(self, delai=3.0):
        """Attend (au plus `delai` secondes) que les derniers événements soient partis."""
        fin = time.time() + delai
        while self.url and not self._file.empty() and time.time() < fin:
            time.sleep(0.05)
        time.sleep(0.1 if self.url else 0)

    def _boucle(self):
        while True:
            evenement = self._file.get()
            donnees = json.dumps(evenement).encode("utf-8")
            for essai in range(10):  # le tableau de bord peut démarrer quelques secondes après
                try:
                    requete = urllib.request.Request(self.url, data=donnees,
                                                     headers={"Content-Type": "application/json"})
                    self._ouvreur.open(requete, timeout=2).read()
                    break
                except Exception:
                    time.sleep(0.5)  # le tableau de bord est facultatif : on abandonne après 10 essais


def apercu_octets(blocs, nb=96):
    """Les premiers octets reçus, en hexadécimal : ce que le serveur « voit »."""
    if not blocs:
        return ""
    return bytes(blocs[0][:nb]).hex()
