"""Vérifie que toutes les bibliothèques du mémoire sont installées
et que les trois schémas de chiffrement fonctionnent."""
import sys
import time

import numpy as np
import pandas as pd
import phe
import psutil
import sklearn
import tenseal as ts
import torch
from phe import paillier

print("=== Versions ===")
print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__, "| GPU disponible :", torch.cuda.is_available())
print("TenSEAL     :", ts.__version__)
print("phe         :", phe.__version__)
print("scikit-learn:", sklearn.__version__)
print("pandas      :", pd.__version__)
print("RAM totale  :", round(psutil.virtual_memory().total / 1e9, 1), "Go")

# --- CKKS : addition de deux vecteurs décimaux ---
debut = time.perf_counter()
ctx_ckks = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384,
                      coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx_ckks.global_scale = 2 ** 40
a = ts.ckks_vector(ctx_ckks, [0.10, 0.25, -0.50])
b = ts.ckks_vector(ctx_ckks, [0.20, 0.25, 0.10])
resultat = (a + b).decrypt()
print("\nCKKS     :", np.round(resultat, 4), "(attendu : [0.3, 0.5, -0.4])",
      f"- {time.perf_counter() - debut:.2f} s")

# --- BFV : addition de deux vecteurs entiers ---
debut = time.perf_counter()
ctx_bfv = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192,
                     plain_modulus=1032193)
a = ts.bfv_vector(ctx_bfv, [1, 2, 3])
b = ts.bfv_vector(ctx_bfv, [10, 20, 30])
print("BFV      :", (a + b).decrypt(), "(attendu : [11, 22, 33])",
      f"- {time.perf_counter() - debut:.2f} s")

# --- Paillier : clé de 3072 bits (128 bits de sécurité) ---
debut = time.perf_counter()
cle_publique, cle_privee = paillier.generate_paillier_keypair(n_length=3072)
somme = cle_publique.encrypt(0.10) + cle_publique.encrypt(0.25)
print("Paillier :", round(cle_privee.decrypt(somme), 4), "(attendu : 0.35)",
      f"- {time.perf_counter() - debut:.2f} s")

print("\nTout fonctionne. Prêt pour le mémoire !")