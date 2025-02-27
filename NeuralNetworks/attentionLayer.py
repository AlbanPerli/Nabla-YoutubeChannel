import math
from activationLayers import Softmax
from abstractLayer import Layer
from denseLayer import Dense
from perceptron import Perceptron

class AttentionLayer(Layer):
    
    def __init__(self, embed_size):
        self.embed_size = embed_size
        self.Q_dense = Dense(embed_size, embed_size)
        self.K_dense = Dense(embed_size, embed_size)
        self.V_dense = Dense(embed_size, embed_size)
        
    def forward(self, sequence):
        
        seq_len = len(sequence)
        
        Qs, Ks, Vs = self.get_QKV(sequence)
        
        output_vectors = []
        attention_scores = []
        
        for Q in Qs:
            score = self.calculate_scores(Q, Ks)
            scaled_scores = self.apply_scaling(score)
            attention_scores.append(scaled_scores)
            attentions_weights = self.apply_softmax(scaled_scores)
            weighted_values = []
            for i in range(len(Vs)):
                weighted_values.append(self.update_V(attentions_weights[i], Vs[i]))
            output_vector = self.extract_output(weighted_values)
            output_vectors.append(output_vector)
        
        return output_vectors, attention_scores
    
    def get_QKV(self, sequence, learnable=False):
        Qs = []
        Ks = []
        Vs = []
        if learnable:
            for i in range(len(sequence)):
                Qs.append(self.Q_dense(f=sequence[i]))
                Ks.append(self.K_dense(f=sequence[i]))
                Vs.append(self.V_dense(f=sequence[i]))
        else:
            for i in range(len(sequence)):
                Qs.append(sequence[i])
                Ks.append(sequence[i])
                Vs.append(sequence[i])
        return Qs, Ks, Vs
    
    def update_V(self,w,V):
        return [w*v for v in V]    
    
    def extract_output(self, weighted_Vs):
        output_vector = [0.0 for _ in range( len(weighted_Vs[0]))]
        for i in range(len(weighted_Vs)):
            for j in range(len(weighted_Vs[i])):
                output_vector[j] += weighted_Vs[i][j]
        return output_vector
            
    def apply_scaling(self, scores):
        return [s/math.sqrt(self.embed_size) for s in scores]
        
    def apply_softmax(self, scores):
        return Softmax()(f=scores)
        
    def calculate_scores(self, Q, Ks):
        scorer = Perceptron(n_inputs=len(Q), weights=Q, use_bias=False)
        scores = []
        for K in Ks:
            scores.append(scorer(f=K))
        return scores
    
    def backward(self, output_grad):
        pass


from test import *

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
    

 

