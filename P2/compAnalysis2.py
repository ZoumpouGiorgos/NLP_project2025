from comp_analyzer import show_comparison_analysis

sentences_txt2 = [
    ("Original", "I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("Vennify", "I believe the team, although there was a bit delay and less communication at recent days, really tried best for paper and cooperation."),
    ("Ateeqq", "The team has encountered a few setbacks and communication challenges lately, yet they have genuinely tried their best for the paper and teamwork."),
    ("Sshleifer", "Despite the fact that bit delay and less communication at recent days, they really tried best for paper and cooperation."),
    ("Chat-GPT", "I believe that although the team has been a bit delayed and communicated less in recent days, they still did their best on the paper and in working together.")
]

show_comparison_analysis(sentences_txt2)