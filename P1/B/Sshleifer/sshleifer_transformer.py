from transformers import AutoTokenizer, logging, AutoModelForSeq2SeqLM

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_summary(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=120,
        min_length=60,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        early_stopping=True,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)