from similarity_matrix_generator import generate_similarity_matrix

sentences_txt1 = [
    ("original", "Today is our dragon boat festival, in our Chinese culture , to celebrate it with all safe and great in our lives."),
    ("custom", "It is the day of our dragon boat festival, as part of our Chinese tradition , to celebrate it while making sure all aspects of our lives are safe and great."),
    ("vennify", "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."),
    ("sshleifer", "Chinese culture celebrates dragon boat festival with all safe and great in our Chinese lives."),
    ("ateeqq", "Our dragon boat festival (in Chinese culture) celebrates today with all safety and great in our lives."),
    ("Chat-GPT", "Today is the Dragon Boat Festival, a cherished tradition in Chinese culture, celebrated with wishes for safety and prosperity in our lives.") # Ground Truth - The best possible model to use
]

print(f"\n{generate_similarity_matrix(sentences_txt1)}\n")