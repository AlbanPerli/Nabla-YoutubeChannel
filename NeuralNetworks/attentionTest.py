import numpy as np
import pandas as pd

from layers.attentionLayer import AttentionLayer

# Dictionnaires de conversion des catégories en valeurs numériques
categories = {
    "animal": 0.1, "objet": 0.2, "humain": 0.3, "action": 0.4,
    "lieu": 0.5, "émotion": 0.6, "nature": 0.7, "temps": 0.8, "nombre": 0.9, "autre": 1.0
}

genres = {
    "masculin": 0.1, "féminin": 0.2, "neutre": 0.3
}

types = {
    "nom": 0.1, "verbe": 0.2, "adjectif": 0.3, "adverbe": 0.4,
    "pronom": 0.5, "préposition": 0.6, "conjonction": 0.7, "autre": 0.8
}

frequence = {
    "rare": 0.1, "normal": 0.5, "très fréquent": 0.9
}

def get_word_vector(word, category, gender, word_type, frequency):
    """
    Génère un vecteur de 4 valeurs pour représenter un mot.

    :param word: Le mot à encoder (non utilisé directement)
    :param category: Catégorie du mot (ex: "animal", "objet", "action")
    :param gender: Genre du mot (ex: "masculin", "féminin", "neutre")
    :param word_type: Type grammatical du mot (ex: "nom", "verbe", "adjectif")
    :param frequency: Fréquence d'utilisation (ex: "rare", "normal", "très fréquent")
    :return: Vecteur [Val1, Val2, Val3, Val4]
    """

    val1 = categories.get(category.lower(), 1.0)  # Par défaut "autre"
    val2 = genres.get(gender.lower(), 0.3)  # Par défaut "neutre"
    val3 = types.get(word_type.lower(), 0.8)  # Par défaut "autre"
    val4 = frequence.get(frequency.lower(), 0.5)  # Par défaut "normal"

    return [val1, val2, val3, val4]

# Ajout de la gestion des mots de la phrase suivante :
# "Ce matin, Jean est allé à cheval au 6 rue lumière."


words_to_process = [
    ("Acheter", "action", "neutre", "verbe", "très fréquent"),
    ("une", "autre", "féminin", "pronom", "très fréquent"),
    ("maison", "objet", "féminin", "nom", "très fréquent"),
    ("à", "autre", "neutre", "préposition", "très fréquent"),
    ("un", "autre", "masculin", "pronom", "très fréquent"),
    ("prix", "objet", "masculin", "nom", "très fréquent"),
    ("raisonnable", "autre", "neutre", "adjectif", "normal")
]

# Générer les vecteurs pour chaque mot de la phrase
word_vectors_extended = {word[0]: get_word_vector(*word) for word in words_to_process}

print(word_vectors_extended)
seq = []
words = []
for k, v in word_vectors_extended.items():
    seq.append(v)
    words.append(k)
    
if __name__ == "__main__":
    att = AttentionLayer(3)
    #seq = [[0.7, 0.2, 0.9], [0.7, 0.1,0.2], [0.3, 0.7, 0.8]]
    weighted_seq, attention_score = att.forward(sequence=seq)
    print("Attention scores: ")
    for a in attention_score:
        print(a)
    print("Weighted sequence: ")
    for ws in weighted_seq:
        print(ws)
        
    df_extended_word_vectors = pd.DataFrame.from_dict(word_vectors_extended, orient="index", columns=["Catégorie", "Genre", "Type", "Fréquence"])
    print(df_extended_word_vectors)
    # print a dataframe with words and their attention scores (confusion matrix)
    df_attention_scores = pd.DataFrame(attention_score, index=words, columns=words)
    print(df_attention_scores)
    

 