from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

app = Flask(__name__)

# Load model from Hugging Face
MODEL_NAME = "Yashasri-04/hate-detection"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_message(text):
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

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return "abusive" if prediction == 1 else "non-abusive"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message", "")

    result = predict_message(message)

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
