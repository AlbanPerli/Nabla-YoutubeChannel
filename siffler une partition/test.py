import networkx as nx
import matplotlib.pyplot as plt
import random

# Données simulées : mots suggérés par un LLM avec probabilités
def generer_arbre_proba(profondeur=3, largeur=3):
    """
    Génère un arbre probabiliste simulant la sortie d'un LLM.
    Chaque nœud est un mot et les arêtes représentent des probabilités.
    """
    G = nx.DiGraph()
    racine = "Début"
    G.add_node(racine)

    def ajouter_noeuds(parent, niveau):
        if niveau >= profondeur:
            return
        
        for i in range(largeur):
            mot = f"Mot{niveau+1}-{i+1}"
            proba = round(random.uniform(0.1, 0.9), 2)  # Probabilité aléatoire
            G.add_node(mot)
            G.add_edge(parent, mot, weight=proba)
            ajouter_noeuds(mot, niveau + 1)

    ajouter_noeuds(racine, 0)
    return G

def afficher_arbre(G):
    """
    Affiche graphiquement l'arbre avec NetworkX et Matplotlib.
    """
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # Disposition en arbre
    labels = {edge: G.edges[edge]['weight'] for edge in G.edges}
    
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    
    plt.title("Arbre de Probabilités généré par un LLM")
    plt.show()

# Générer et afficher l'arbre
arbre = generer_arbre_proba()
afficher_arbre(arbre)
