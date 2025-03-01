from graphviz import Digraph
from denseLayer import Dense
from layers import CNN1D, Module, Parallel
from perceptron import Perceptron

def visualize_module(module, input_size):
    """
    Génère un graphe de la structure du module avec les connexions entre couches et neurones.
    
    :param module: Un objet de type Module, Dense, Perceptron ou CNN1D
    :param input_size: Nombre d'entrées
    """
    dot = Digraph(format="png")
    dot.attr(rankdir="LR", splines="line", nodesep="1.0", ranksep="1.3")  

    # Paramètres de style pour uniformiser la taille des nœuds
    node_size = {"width": "1.0", "height": "1.0", "fixedsize": "true"}

    # Création des nœuds d'entrée (avec taille uniforme)
    for i in range(input_size):
        dot.node(f"Input{i}", f"X{i}", shape="square", style="filled", fillcolor="lightgray", **node_size)

    previous_layer = [f"Input{i}" for i in range(input_size)]
    layer_id = 0

    # Si l'objet n'est pas un Module, le transformer en liste de couches
    if isinstance(module, (Dense, Perceptron, CNN1D, Parallel)):
        layers = [module]
    elif isinstance(module, Module):
        layers = module.layers
    else:
        raise ValueError("Type de module non pris en charge.")

    # Parcours des couches
    for layer in layers:
        layer_name = f"Layer{layer_id}"
        
        if isinstance(layer, Dense):
            # Création des perceptrons de la couche Dense
            neurons = [f"{layer_name}_N{i}" for i in range(len(layer.neurons))]
            for neuron in neurons:
                dot.node(neuron, shape="circle", style="filled", fillcolor="lightblue", **node_size)
            # Connexion avec la couche précédente
            for prev in previous_layer:
                for neuron in neurons:
                    dot.edge(prev, neuron)
            previous_layer = neurons
        
        elif isinstance(layer, CNN1D):
            # Création des filtres
            filters = [f"{layer_name}_F{i}" for i in range(len(layer.filters))]
            for f in filters:
                dot.node(f, shape="diamond", style="filled", fillcolor="lightgreen", **node_size)
            # Connexion avec la couche précédente
            for prev in previous_layer:
                for f in filters:
                    dot.edge(prev, f)
            previous_layer = filters
            
        elif isinstance(layer, Perceptron):
            # Création d'un seul neurone
            neuron = f"{layer_name}_N0"
            dot.node(neuron, shape="circle", style="filled", fillcolor="green", **node_size)
            for prev in previous_layer:
                dot.edge(prev, neuron)
            previous_layer = [neuron]
        
        elif isinstance(layer, Parallel):
            # Gestion des couches parallèles
            sub_outputs = []
            for sub_layer in layer.layers:
                sub_name = f"{layer_name}_Sub{layer.layers.index(sub_layer)}"
                if isinstance(sub_layer, Dense):
                    neurons = [f"{sub_name}_N{i}" for i in range(len(sub_layer.neurons))]
                    for neuron in neurons:
                        dot.node(neuron, shape="circle", style="filled", fillcolor="lightblue", **node_size)
                    for prev in previous_layer:
                        for neuron in neurons:
                            dot.edge(prev, neuron)
                    sub_outputs.extend(neurons)
                elif isinstance(sub_layer, Perceptron):
                    neuron = f"{sub_name}_N0"
                    dot.node(neuron, shape="circle", style="filled", fillcolor="green", **node_size)
                    for prev in previous_layer:
                        dot.edge(prev, neuron)
                    sub_outputs.append(neuron)
                elif isinstance(sub_layer, CNN1D):
                    filters = [f"{sub_name}_F{i}" for i in range(len(sub_layer.filters))]
                    for f in filters:
                        dot.node(f, shape="diamond", style="filled", fillcolor="lightgreen", **node_size)
                    for prev in previous_layer:
                        for f in filters:
                            dot.edge(prev, f)
                    sub_outputs.extend(filters)
            previous_layer = sub_outputs
        
        layer_id += 1

    # Création des nœuds de sortie (avec taille uniforme)
    output_nodes = [f"Output{i}" for i in range(len(previous_layer))]
    for output, prev in zip(output_nodes, previous_layer):
        dot.node(output, shape="square", style="filled", fillcolor="lightcoral", **node_size)
        dot.edge(prev, output)

    # Affichage du graphe
    return dot

# ---- TESTS ----

# Test avec un module complet
module = Module(Dense(8, 3, lr=0.1),
                Dense(3, 8, lr=0.1),
                Parallel(Dense(8, 8, lr=0.1),
                         Perceptron(8)))
dot = visualize_module(module, input_size=8)
dot.render("module_structure", view=True)  # Sauvegarde et ouvre l'image

# Test avec un seul Dense()
dot = visualize_module(Dense(4, 8, lr=0.1), input_size=4)
dot.render("dense_structure", view=True)

# Test avec un seul Perceptron()
dot = visualize_module(Perceptron(4), input_size=4)
dot.render("perceptron_structure", view=True)
