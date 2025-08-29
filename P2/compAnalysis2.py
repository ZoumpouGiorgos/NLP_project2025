from comp_analyzer import show_comparison_analysis

sentences_txt2 = [
    ("Original",
     "During our final discuss, I told him about the new submission — the one we were waiting since\n"
     "last autumn, but the updates was confusing as it not included the full feedback from reviewer or\n"
     "maybe editor?\n"
     "Anyway, I believe the team, although bit delay and less communication at recent days, they really\n"
     "tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance\n"
     "and efforts until the Springer link came finally last week, I think.\n"
     "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before\n"
     "he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.\n"
     "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future\ntargets"),
    ("Vennify",
     "During our final discussion, I told him about the new submission — the one we were waiting for since then. Last autumn, the updates were confusing as they did not include the full feedback from reviewer or reviewer. Maybe an editor? Anyway, I believe the team, although there was a bit of delay and less communication at recent days, they really did have a good time. We should be grateful, I mean all of us, for the acceptance. And efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plans for the acknowledgments section edit before the appointment. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future. Targets are the correct targets."),
    ("Ateeqq",
     "At our final talk, I told him that we were waiting for the new submission — the one we were waiting for last autumn, but the updates were confusing as it did not include the full feedback from reviewer or perhaps editor? Despite the fact that bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me that the doctor plan for the acknowledgments section edit."),
    ("Sshleifer",
     "The Springer link came finally last week, I think, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation . We should be grateful for the acceptance efforts ."),
    ("Chat-GPT",
     "During our final discussion, I mentioned the new submission—the one we had been waiting for since last autumn. However, the update was a bit unclear, as it didn’t seem to include the complete feedback from the reviewer, or perhaps the editor.\n"
     "In any case, I believe the team—despite some delays and less communication in recent days—really did their best with the paper and our collaboration. We should all be grateful for the acceptance and the effort put in, until the Springer link was finally available last week.\n"
     "Also, please remind me if the doctor still plans to revise the acknowledgments section before sending it again, as I haven’t seen the final version of that part yet. If I missed it, I apologize.\n"
     "Overall, let’s make sure everything is in order, celebrate this achievement with a strong coffee, and look forward to our future goals.")
]

show_comparison_analysis(sentences_txt2)