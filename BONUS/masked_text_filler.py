from transformers import pipeline, logging
logging.set_verbosity_error()

import nltk
nltk.download('punkt')

def print_filled_masked_text(text,model_id,mask_token):
    try:
        pipe = pipeline("fill-mask", model=model_id)
    except Exception as e:
        print(f"{model_id} was not loaded: {e}")
        return None

    print(f"Input:\n{text}\n")
    filled_text = text
    try:
        preds = pipe(text, top_k=1)
        if isinstance(preds, list) and isinstance(preds[0], list):
            temp_text = text
            for mask_preds in preds:
                best = mask_preds[0]['token_str']
                temp_text = temp_text.replace(mask_token, best, 1)
            filled_text = temp_text
        else:
            best = preds[0]['token_str']
            filled_text = text.replace(mask_token, best, 1)
            
        print(f"{model_id} filled text:\n{filled_text}\n")
    except Exception as e:
        print(f"{model_id} error:", e)