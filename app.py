from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)

# Load model once at startup
model_path = "./hate_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


# 🔹 NEW: Text Cleaning Function
def clean_text(text):
    text = text.lower()  # convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text


def predict_message(text):
    # 🔹 Clean text before prediction
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    return "abusive" if prediction == 1 else "non-abusive"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message")

    result = predict_message(message)

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)