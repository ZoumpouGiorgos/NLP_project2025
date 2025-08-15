from comp_analyzer import show_comparison_analysis

sentences_txt1 = [
    ("Original",
     "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. "
     "Thank your message to show our words to the doctor, as his next contract checking, to all of us. "
     "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. "
     "I am very appreciated the full support of the professor, for our Springer proceedings publication."),
    ("Vennify",
     "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. "
     "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago."),
    ("Ateeqq",
     "We celebrate today's dragon boat festival (in our Chinese culture) to celebrate it with all safety and great in our lives. Hope you too to enjoy it as my deepest wishes. "
     "Thank you message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. "
     "I have received the message from the professor, to show me, this, a couple of days ago. I am very much appreciated the full support of the professor, for our Springer proceedings publication."),
    ("Sshleifer",
     "It is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in their lives. "
     "I am very appreciated the full support of the Professor, for our Springer proceedings publication. "
     "We are happy to show our words to the doctor, as his next contract checking to all of us."),
    ("Chat-GPT",
     "Today is the Dragon Boat Festival, a cherished celebration in Chinese culture, symbolizing safety, prosperity, and well-being. I hope you also enjoy this occasion, along with my heartfelt wishes. "
     "Thank you for passing on our words to the doctor regarding his upcoming contract review on behalf of all of us. I’ve seen the approved message—actually, the professor had already shared it with me a couple of days ago. "
     "I truly appreciate the professor’s full support for our Springer proceedings publication.")
]

show_comparison_analysis(sentences_txt1)