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