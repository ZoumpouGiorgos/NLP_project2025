from masked_text_filler import print_filled_masked_text

texts = [
    "Αν η κυριότητα του <mask> ανήκει σε περισσότερους <mask> αδιαιρέτου κατ΄ιδανικά <mask>, εφαρμόζονται οι διατάξεις για την κοινωνία.",
    "Στο κοινό <mask> μπορεί να συσταθεί πραγματική δουλεία υπέρ του <mask> κύριου άλλου ακινήτου και αν ακόμη αυτός είναι <mask> του ακινήτου που βαρύνεται με τη δουλεία."
]

print_filled_masked_text(texts[0],"xlm-roberta-base","<mask>")
print_filled_masked_text(texts[1],"xlm-roberta-base","<mask>")