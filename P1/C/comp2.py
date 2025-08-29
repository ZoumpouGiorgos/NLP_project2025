from similarity_matrix_generator import generate_similarity_matrix

sentences_txt2 = [
    ("original", "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("custom", "It is clear the team, even though there have been some delays and reduced communication recently, they genuinely gave their best effort on the manuscript and collaboration."),
    ("vennify", " I believe the team, although there was a bit delay and less communication at recent days, really tried best for paper and cooperation."),
    ("sshleifer", " believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("ateeqq", "Despite the fact that bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("Chat-GPT", "I believe that although the team has been a bit delayed and communicated less in recent days, they still did their best on the paper and in working together.") # Ground Truth - The best possible model to use
]

print(f"Text2 Similarity Matrix:\n{generate_similarity_matrix(sentences_txt2)}\n")