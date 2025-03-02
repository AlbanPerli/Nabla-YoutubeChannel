from layers.activationLayers import SigmoidLayer
from layers.denseLayer import Dense
from layers.layers import Module


module = Module(Dense(5, 32, lr=0.1),
                SigmoidLayer(32),
                Dense(32, 32, lr=0.1),
                SigmoidLayer(32),
                Dense(32, 5, lr=0.1),
                SigmoidLayer(5))

import numpy as np
import random
import string

def generate_seq2seq_dataset(num_samples=8, seq_length=5):
    """
    Génère un dataset seq2seq où chaque entrée est une séquence et la sortie est la même séquence décalée d'un pas.
    
    Args:
        num_samples (int): Nombre d'exemples à générer.
        seq_length (int): Longueur de chaque séquence.
    
    Returns:
        list of tuples: Liste contenant (entrée, sortie) pour chaque exemple.
    """
    dataset = []
    
    for _ in range(num_samples):
        input_seq = [random.uniform(0, 1) for _ in range(seq_length)]
        output_seq = input_seq[1:] + [random.uniform(0, 1)]  # Décalage à gauche avec un nouvel élément
        dataset.append((input_seq, output_seq))
    
    return dataset

# Exemple d'utilisation
dataset = generate_seq2seq_dataset(num_samples=8, seq_length=5)

    
# Test avec un vecteur d'entrée
for i in range(1000):
    for x, y_exp in dataset:
        y = module(f=x)
        errors = [e - y[i] for i, e in enumerate(y_exp)]
        out_grad = module(b=errors)
    

for x, y_exp in dataset:
    y = module(f=x)
    print(f"Entrée : {x}\n -> Sortie : {y}\n (Attendu : {y_exp})")