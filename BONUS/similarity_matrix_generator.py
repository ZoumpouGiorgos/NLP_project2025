from sentence_transformers import SentenceTransformer, util
import pandas as pd

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def generate_similarity_matrix(sentences):
    labels, texts = zip(*sentences)
    embeddings = model.encode(texts, convert_to_tensor=True)

    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    df = pd.DataFrame(similarity_matrix.cpu().numpy(), index=labels, columns=labels).round(2)

    return df

sentences_a1113 = [
    ("original",         ['πράγματος','εξ','μέρη']     ),
    ("nlpaueb",          ['ακινητου','εξ','μερη']      ),
    ("xlm-roberta-base", ['κράτους','του','δικαιώματα'])
]

sentences_a1114 = [
    ("original",         ['βάρος','ακίνητο','εκάστοτε','συγκύριος','πραγματική','κάποιος','ακινήτου']  ),
    ("nlpaueb",          ['ακινητο','ακινητο','εκαστοτε','κυριος','πραγματική','καθενας','ακινητου']),
    ("xlm-roberta-base", ['σπίτι','δεν','ενός','κύριο','πραγματική','ένας','κοινού']                )
]

for i in range(3):
    word_masks = [(sentences_a1113[k][0],sentences_a1113[k][1][i]) for k in range(len(sentences_a1113))]


    print(f"A1113 mask no.{i+1}:")
    for m in range(3):
        print(f"-> {sentences_a1113[m][0]}: {sentences_a1113[m][1][i]}")

    print(f"Similarity matrix:")
    print(f"\n{generate_similarity_matrix(word_masks)}\n")

    print("------------------------------------------------------\n")

for i in range(7):
    word_masks = [(sentences_a1114[k][0],sentences_a1114[k][1][i]) for k in range(len(sentences_a1114))]

    print(f"A1114 mask no.{i+1}:")
    for m in range(3):
        print(f"-> {sentences_a1114[m][0]}: {sentences_a1114[m][1][i]}")

    print(f"Similarity matrix:")
    print(f"\n{generate_similarity_matrix(word_masks)}\n")

    print("------------------------------------------------------\n")