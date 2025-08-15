from comp_analyzer import show_comparison_analysis

sentences_txt1 = [
    ("Original", "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."),
    ("Vennify", "Our dragon boat festival (in Chinese culture) celebrates today with all safety and great in our lives."),
    ("Ateeqq", "Chinese culture celebrates dragon boat festival with all safe and great in our Chinese lives."),
    ("Sshleifer", "It is the day of our dragon boat festival, within our rich Chinese heritage, to enjoy it while ensuring everything in our lives is safe and wonderful."),
    ("Chat-GPT", "Today is the Dragon Boat Festival, a cherished tradition in Chinese culture, celebrated with wishes for safety and prosperity in our lives.")
]

show_comparison_analysis(sentences_txt1)