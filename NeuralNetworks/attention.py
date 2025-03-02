# Définition des vecteurs pour une phrase exemple : "Le chat mange un poisson"
# On représente chaque mot par un vecteur de dimension 4

# Query (Q) - Ce qu'on cherche
Q = [0.7, 0.7, 0.2, 0.6]

# Keys (K) - Les représentations des mots
K1 = [0.5, 0.1, 0.9, 0.7]  # "Le"
K2 = [0.7, 0.7, 0.2, 0.6]  # "chat"
K3 = [0.8, 0.2, 0.5, 0.4]  # "mange"
K4 = [0.4, 0.5, 0.3, 0.2]  # "un"
K5 = [0.6, 0.3, 0.7, 0.9]  # "poisson"

# Values (V) - Les informations associées aux mots
V1 = [0.5, 0.1, 0.9, 0.7]  # "Le"
V2 = [0.7, 0.7, 0.2, 0.6]  # "chat"
V3 = [0.8, 0.2, 0.5, 0.4]  # "mange"
V4 = [0.4, 0.5, 0.3, 0.2]  # "un"
V5 = [0.6, 0.3, 0.7, 0.9]  # "poisson"

# Fonction de produit scalaire
def dot_product(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

# Calcul des scores d'attention
scores = [
    dot_product(Q, K1),  # Score avec "Le"
    dot_product(Q, K2),  # Score avec "chat"
    dot_product(Q, K3),  # Score avec "mange"
    dot_product(Q, K4),  # Score avec "un"
    dot_product(Q, K5),  # Score avec "poisson"
]

# Softmax
from layers.activationLayers import *

attention_weights = SoftmaxLayer()(scores)

# Calcul du vecteur pondéré (somme pondérée des valeurs)
output_vector = [0] * 4  # Vecteur de sortie de dimension 4
values = [V1, V2, V3, V4, V5]

for i in range(len(values)):  # Pour chaque valeur V
    for j in range(4):  # Pour chaque dimension
        output_vector[j] += attention_weights[i] * values[i][j]

# Affichage des résultats
import pandas as pd
df = pd.DataFrame({
    "Mot": ["Le", "Chat", "Mange", "Un", "Poisson"],
    "Score": scores,
    "Poids d'attention": attention_weights,
    "Valeur (V)": values
})

print(df)
print("\nVecteur de sortie:", output_vector)

