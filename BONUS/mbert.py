from masked_text_filler import print_filled_masked_text

texts = [
    "Κοινό πράγμα. - Aν η κυριότητα του [MASK] ανήκει σε περισσότερους [MASK] αδιαιρέτου κατ΄ιδανικά [MASK], εφαρμόζονται οι διατάξεις για την κοινωνία.",
    "Πραγματική δουλεία σε [MASK] η υπέρ του κοινού ακινήτου. — Στο κοινό [MASK] μπορεί να συσταθεί πραγματική δουλεία υπέρ του [MASK] κύριου άλλου ακινήτου και αν ακόμη αυτός είναι [MASK] του ακινήτου που βαρύνεται με τη δουλεία. Το ίδιο ισχύει και για την [MASK] δουλεία πάνω σε ακίνητο υπέρ των εκάστοτε κυρίων κοινού ακινήτου, αν [MASK] από αυτούς είναι κύριος του [MASK] που βαρύνεται με τη δουλεία."
]

print_filled_masked_text(texts[0],"bert-base-multilingual-cased","[MASK]")
print_filled_masked_text(texts[1],"bert-base-multilingual-cased","[MASK]")