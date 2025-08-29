from masked_text_filler import print_filled_masked_text

texts = [
    "Αν η κυριότητα του [MASK] ανήκει σε περισσότερους [MASK] αδιαιρέτου κατ΄ιδανικά [MASK], εφαρμόζονται οι διατάξεις για την κοινωνία.",
    "Στο κοινό [MASK] μπορεί να συσταθεί πραγματική δουλεία υπέρ του [MASK] κύριου άλλου ακινήτου και αν ακόμη αυτός είναι [MASK] του ακινήτου που βαρύνεται με τη δουλεία."
]

print_filled_masked_text(texts[0],"nlpaueb/bert-base-greek-uncased-v1","[MASK]")
print_filled_masked_text(texts[1],"nlpaueb/bert-base-greek-uncased-v1","[MASK]")