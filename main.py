from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    return jsonify({"response": f"Received: {data}"})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello from Flask market-analyzer!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

