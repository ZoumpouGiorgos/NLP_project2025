# NLP-Project

## Περιγραφή / Description

Το έργο αυτό υλοποιεί και συγκρίνει διάφορες μεθόδους παραφραστικής επεξεργασίας και ανάλυσης κειμένου, αξιοποιώντας σύγχρονα μοντέλα NLP (Natural Language Processing). Περιλαμβάνει pipelines για παραγωγή παραφράσεων, συνοψίσεων, καθώς και μετρικές σύγκρισης και οπτικοποίησης ομοιότητας μεταξύ διαφορετικών εκδοχών κειμένων.

This project implements and compares various paraphrasing and text analysis methods, utilizing state-of-the-art NLP (Natural Language Processing) models. It includes pipelines for paraphrase and summary generation, as well as similarity metrics and visualization between different text versions.

## Δομή Φακέλων / Folder Structure

- **P1/**
  - **A/**: Custom παραφράσεις με απλούς αλγορίθμους αντικατάστασης λέξεων ([customText1.py](P1/A/customText1.py), [customText2.py](P1/A/customText2.py))
    / Custom paraphrasing with simple word replacement algorithms
  - **B/**: Παραφράσεις/συνοψίσεις με έτοιμα μοντέλα transformers
    / Paraphrasing/summarization with pretrained transformer models
    - **Ateeqq/**: [Ateeqq/Text-Rewriter-Paraphraser](P1/B/Ateeqq/ateeqq_transformer.py)
    - **Sshleifer/**: [sshleifer/distilbart-cnn-12-6](P1/B/Sshleifer/sshleifer_transformer.py)
    - **Vennify/**: [vennify/t5-base-grammar-correction](P1/B/Vennify/vennify_transformer.py)
  - **C/**: Υπολογισμός similarity matrices μεταξύ των παραγόμενων κειμένων ([similarity_matrix_generator.py](P1/C/similarity_matrix_generator.py), [comp1.py](P1/C/comp1.py), [comp2.py](P1/C/comp2.py))
    / Calculation of similarity matrices between generated texts

- **P2/**
  - Προχωρημένη ανάλυση ομοιότητας και οπτικοποίηση embeddings με χρήση BERT, GloVe, FastText ([comp_analyzer.py](P2/comp_analyzer.py), [compAnalysis1.py](P2/compAnalysis1.py), [compAnalysis2.py](P2/compAnalysis2.py))
    / Advanced similarity analysis and embedding visualization using BERT, GloVe, FastText

## Τεχνολογίες & Εξαρτήσεις / Technologies & Dependencies

- Python 3.10+
- PyTorch, Transformers, SentenceTransformers
- Gensim, FastText, NLTK, scikit-learn, pandas, matplotlib
- Τα requirements ορίζονται στο [pyproject.toml](pyproject.toml) / Requirements are defined in [pyproject.toml](pyproject.toml)

## Οδηγίες Εκτέλεσης / Usage

1. **Εγκατάσταση εξαρτήσεων / Install dependencies**
   ```sh
   pip install poetry
   poetry install
   ```

2. **Κατέβασμα προεκπαιδευμένων μοντέλων / Download pretrained models**
   Τα μοντέλα κατεβαίνουν αυτόματα κατά το πρώτο τρέξιμο των scripts.
   Models are downloaded automatically on first script run.

3. **Εκτέλεση παραδειγμάτων / Run examples**
   - Για custom παραφράσεις / For custom paraphrasing:
     ```sh
     python P1/A/customText1.py
     python P1/A/customText2.py
     ```
   - Για paraphrasing/summarization με transformers / For paraphrasing/summarization with transformers:
     ```sh
     python P1/B/Vennify/vennifyTXT1.py
     python P1/B/Ateeqq/ateeqqTXT2.py
     # etc.
     ```
   - Για similarity matrices / For similarity matrices:
     ```sh
     python P1/C/comp1.py
     python P1/C/comp2.py
     ```
   - Για ανάλυση embeddings & οπτικοποίηση / For embedding analysis & visualization:
     ```sh
     python P2/compAnalysis1.py
     python P2/compAnalysis2.py
     ```

## Περιγραφή Λειτουργικότητας / Functionality

- **Custom Paraphrasing**: Απλή αντικατάσταση φράσεων με τυχαίες επιλογές ([P1/A](P1/A/)).
  / Simple phrase replacement with random choices
- **Paraphrasing/Summarization με Transformers**: Χρήση έτοιμων μοντέλων για παραγωγή παραφράσεων ή συνοψίσεων ([P1/B](P1/B/)).
  / Using pretrained models for paraphrasing or summarization
- **Similarity Matrices**: Υπολογισμός ομοιότητας μεταξύ όλων των εκδοχών με Sentence Transformers ([P1/C](P1/C/)).
  / Similarity calculation between all versions using Sentence Transformers
- **Προχωρημένη Ανάλυση**: Υπολογισμός embeddings με BERT, GloVe, FastText, μετρικές ομοιότητας και οπτικοποίηση σε 2D χώρο ([P2](P2/)).
  / Embedding calculation with BERT, GloVe, FastText, similarity metrics and 2D visualization

## Αρχεία Δεδομένων / Data Files

- **glove.6B.300d.txt**: GloVe pre-trained vectors
- **cc.en.300.bin**: FastText pre-trained vectors

---


