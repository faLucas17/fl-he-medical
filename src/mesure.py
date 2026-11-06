"""Mesure de la mémoire vive (RAM) utilisée par un programme.

Un petit fil d'exécution (« thread ») relève la mémoire du processus toutes
les 50 millisecondes et garde la valeur la plus haute : c'est le pic de RAM.

Utilisation :
    with PicMemoire() as memoire:
        ... calcul ...
    print(memoire.pic_mo)      # pic total du processus, en mégaoctets
    print(memoire.ajout_mo)    # mémoire ajoutée par le calcul (pic - niveau de départ)

Le pic total comprend les bibliothèques déjà chargées (PyTorch, TenSEAL...),
soit environ 700 Mo. La mémoire ajoutée isole le coût du calcul lui-même.
"""
import threading

import psutil


class PicMemoire:
    def __init__(self, intervalle=0.05):
        self.intervalle = intervalle
        self.processus = psutil.Process()
        self.pic_octets = 0
        self._arret = threading.Event()
        self._fil = threading.Thread(target=self._surveiller, daemon=True)

    def _surveiller(self):
        while not self._arret.is_set():
            self.pic_octets = max(self.pic_octets, self.processus.memory_info().rss)
            self._arret.wait(self.intervalle)

    def __enter__(self):
        self.depart_octets = self.processus.memory_info().rss
        self.pic_octets = self.depart_octets
        self._fil.start()
        return self

    def __exit__(self, *erreur):
        self._arret.set()
        self._fil.join()
        self.pic_octets = max(self.pic_octets, self.processus.memory_info().rss)

    @property
    def pic_mo(self):
        return self.pic_octets / 1024 ** 2

    @property
    def ajout_mo(self):
        return (self.pic_octets - self.depart_octets) / 1024 ** 2
