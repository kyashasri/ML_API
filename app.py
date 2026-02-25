from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/Yashasri-04/hate-detection"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

def predict_message(text):
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()

    # Hugging Face returns scores list
    try:
        label = result[0][0]["label"]
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
