from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask_cors import CORS
import string
import random

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "").strip()

    if input_text == "":
        return jsonify({"prediction": ""})

    # Encode input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True)

    # Generate multiple tokens
    max_gen_tokens = 20  # generate enough tokens for a few words
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_gen_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove original input
    continuation = generated_text[len(input_text):].strip()

    # Remove leading punctuation
    continuation = continuation.lstrip(string.punctuation + " ")

    # Split into words and take 2–5 words
    words = [w for w in continuation.split() if any(c.isalnum() for c in w)]
    num_words = random.randint(2, 5)  # randomly choose 2–5 words for variety
    next_words_text = " ".join(words[:num_words]) if words else ""

    return jsonify({"prediction": next_words_text})

if __name__ == "__main__":
    app.run(debug=True)
