from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Ateeqq/Text-Rewriter-Paraphraser")
model = AutoModelForSeq2SeqLM.from_pretrained("Ateeqq/Text-Rewriter-Paraphraser")

def generate_paraphrase(text):
    # Add prefix as expected by the model
    input_text = "paraphraser: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    # Generate paraphrases using beam search
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=6,
        num_return_sequences=3,
        num_beam_groups=3,
        diversity_penalty=2.0,
        early_stopping=True,
        trust_remote_code=True
    )
    
    # Decode and return paraphrased text
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[-1]