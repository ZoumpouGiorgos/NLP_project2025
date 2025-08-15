from naive_sentence_reconstructor import reconstruct_sentence

text2 = "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

text2_replacements = {
        "I believe the team": [
            "I am confident that the team",
            "It is clear that the team"
        ],
        "although bit delay and less communication at recent days": [
            "even though there have been some delays and reduced communication recently",
            "despite recent slow progress and less interaction",
            "although communication and progress were a bit slow lately"
        ],
        "they really tried best for paper and cooperation": [
            "they have truly done their best with the paper and teamwork",
            "they genuinely gave their best effort on the manuscript and collaboration",
            "they worked hard and cooperated well on the paper"
        ]
    }

print(f"Text2 Reconstructed Sentence:\n{ reconstruct_sentence(text2,text2_replacements) }\n")