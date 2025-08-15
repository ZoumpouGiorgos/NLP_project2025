from sentence_transformers import SentenceTransformer, util
import pandas as pd

def generate_similarity_matrix(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    labels, texts = zip(*sentences)

    embeddings = model.encode(texts, convert_to_tensor=True)

    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    df = pd.DataFrame(similarity_matrix.cpu().numpy(), index=labels, columns=labels).round(2)

    return df