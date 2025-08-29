import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#PREPROCESSING
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

#EMBEDDING METHODS
def embed_bert(texts, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def embed_avg_word2vec(texts, model):
    vectors = []
    for text in texts:
        tokens = preprocess(text)
        word_vecs = [model[w] for w in tokens if w in model]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            glove_model[word] = vector
            
    vector_size = len(next(iter(glove_model.values())))
    kv = KeyedVectors(vector_size=vector_size)
    kv.add_vectors(list(glove_model.keys()), list(glove_model.values()))
    return kv

def embed_avg_fasttext(texts, ft_model):
    vectors = []
    for text in texts:
        tokens = preprocess(text)
        word_vecs = [ft_model.get_word_vector(w) for w in tokens]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(ft_model.get_dimension()))
    return np.array(vectors)

#SIMILARITY METRICS
def similarity_metrics(emb1, emb2):
    return {
        "cosine": 1 - cosine(emb1, emb2),
        "euclidean": euclidean(emb1, emb2)
    }

#VISUALIZATION WITH ARROWS
def visualize_with_arrows(embeddings, labels, method):
    if method not in ["PCA", "t-SNE"]:
        print("Method must be 'PCA' or 't-SNE'")
        return

    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "t-SNE":
        perplexity = min(len(embeddings) - 1, 5)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

    reduced = reducer.fit_transform(embeddings)
    reduced = (reduced - reduced.mean(0)) / (reduced.std(0) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')

    ax.scatter(reduced[0, 0], reduced[0, 1], label="Original", s=150)
    for i in range(1, len(reduced)):
        ax.scatter(reduced[i, 0], reduced[i, 1], label=labels[i])
        ax.arrow(
            reduced[0, 0], reduced[0, 1],
            reduced[i, 0] - reduced[0, 0],
            reduced[i, 1] - reduced[0, 1],
            alpha=0.5, head_width=0.05, length_includes_head=True
        )

    ax.legend()
    ax.set_title(f"Embedding Shift Visualization ({method})")
    plt.show()

#MAIN ANALYSIS
def show_comparison_analysis(sentences):
    labels, texts = zip(*sentences)

    #BERT
    print("\nBERT ANALYSIS")
    emb_bert = embed_bert(texts)
    for i in range(1, len(emb_bert)):
        metrics = similarity_metrics(emb_bert[0], emb_bert[i])
        print(f"{labels[i]}: Cosine={metrics['cosine']:.4f}, Euclidean={metrics['euclidean']:.4f}")
    visualize_with_arrows(emb_bert, labels, method="PCA")
    visualize_with_arrows(emb_bert, labels, method="t-SNE")

    #GloVe
    print("\nGloVe ANALYSIS")
    glove_model = load_glove_model("glove.6B.300d.txt")
    emb_glove = embed_avg_word2vec(texts, glove_model)
    for i in range(1, len(emb_glove)):
        metrics = similarity_metrics(emb_glove[0], emb_glove[i])
        print(f"{labels[i]}: Cosine={metrics['cosine']:.4f}, Euclidean={metrics['euclidean']:.4f}")
    visualize_with_arrows(emb_glove, labels, method="PCA")
    visualize_with_arrows(emb_glove, labels, method="t-SNE")

    #FastText
    print("\nFastText ANALYSIS")
    import fasttext
    import fasttext.util
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    emb_fasttext = embed_avg_fasttext(texts, ft)
    for i in range(1, len(emb_fasttext)):
        metrics = similarity_metrics(emb_fasttext[0], emb_fasttext[i])
        print(f"{labels[i]}: Cosine={metrics['cosine']:.4f}, Euclidean={metrics['euclidean']:.4f}")
    visualize_with_arrows(emb_fasttext, labels, method="PCA")
    visualize_with_arrows(emb_fasttext, labels, method="t-SNE")