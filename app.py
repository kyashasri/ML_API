from flask import Flask, request, jsonify
import requests
import os
import re

app = Flask(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/Yashasri-04/hate-detection"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_message(text):
    text = clean_text(text)

    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()

    try:
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                label = result[0].get("label", "error")
            elif isinstance(result[0], list) and len(result[0]) > 0:
                label = result[0][0].get("label", "error")
            else:
                label = "error"
        else:
            label = "error"
    except:
        label = "error"

    return label


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message", "")

    result = predict_message(message)
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
