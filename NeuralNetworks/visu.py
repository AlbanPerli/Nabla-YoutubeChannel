import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import ace_tools as tools

def softmax(x):
    """Applique la fonction softmax pour normaliser les scores d'attention."""
    exp_x = np.exp(x - np.max(x))  # Stabilisation numérique
    return exp_x / np.sum(exp_x)

# Définition d'une phrase avec un embedding significatif
words = ["La", "Voiture", "Roule", "Vite", "sur", "une", "route", "de", "campagne"]
sequence = np.array([
    [0.2, 0.1, 0.2],  # "La"        (Déterminant, Statique, Peu important)
    [0.8, 0.5, 0.7],  # "Voiture"   (Objet, Mobile, Important)
    [0.9, 0.9, 0.9],  # "Roule"     (Action, Mobile, Très important)
    [0.4, 0.7, 0.8],  # "Vite"      (Adverbe, Rapide, Moyennement important)
    [0.3, 0.2, 0.4],  # "sur"       (Préposition, Statique, Peu important)
    [0.2, 0.3, 0.2],  # "une"       (Déterminant, Statique, Peu important)
    [0.7, 0.6, 0.5],  # "route"     (Lieu, Mobile, Important)
    [0.3, 0.4, 0.3],  # "de"        (Préposition, Statique, Peu important)
    [0.7, 0.8, 0.7]   # "campagne"  (Lieu, Statique, Important)
])

# En self-attention, Q = K = V
Q_matrix = sequence
K_matrix = sequence
V_matrix = sequence

# Calcul des scores d'attention (Q · K^T)
score_matrix = np.dot(Q_matrix, K_matrix.T)

# Softmax pour obtenir les poids d'attention
attention_matrix = np.array([softmax(scores) for scores in score_matrix])

# Calcul des nouvelles représentations des mots
output_matrix = np.dot(attention_matrix, V_matrix)

# Conversion en DataFrames pour affichage
df_scores = pd.DataFrame(score_matrix, index=words, columns=words)
df_attention = pd.DataFrame(attention_matrix, index=words, columns=words)
df_output = pd.DataFrame(output_matrix, index=words, columns=["Concept", "Mouvement", "Intensité"])
print(df_output)
# Affichage des résultats
# tools.display_dataframe_to_user(name="Scores d'Attention (Avant Softmax)", dataframe=df_scores)
# tools.display_dataframe_to_user(name="Matrice d'Attention (Après Softmax)", dataframe=df_attention)
# tools.display_dataframe_to_user(name="Représentation mise à jour (Self-Attention)", dataframe=df_output)


def plot_attention_matrix(matrix, title):

    plt.rcParams['figure.facecolor'] = 'black'

    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
 
    plt.figure(figsize=(6,6))
    sns.heatmap(matrix, annot=True, cmap="inferno", xticklabels=words, yticklabels=words)
    plt.title(title)
    plt.xlabel("Mots (Keys)")
    plt.ylabel("Mots (Queries)")
    plt.show()
    

plot_attention_matrix(score_matrix, "Scores d'Attention (Avant Softmax)")