from transformers import AutoTokenizer, logging, AutoModelForSeq2SeqLM

model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_paraphrase(text):
    sentences = text.split('\n')
    corrected = []
    for s in sentences:
        input_text = f"correct grammar: {s}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected.append(corrected_text)
    return ' '.join(corrected)