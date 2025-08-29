from transformers import AutoTokenizer, logging, AutoModelForSeq2SeqLM
logging.set_verbosity_error()

model_name = "Ateeqq/Text-Rewriter-Paraphraser"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_paraphrase(text):
    input_text = "paraphraser: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
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
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[-1]