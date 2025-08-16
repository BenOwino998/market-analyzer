from flask import Flask, jsonify, request
from FAA_MVP import run_analysis  # <-- use the entry point we just wrote

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello from Fractal Market Analyzer (Flask)!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True, silent=False)
        result = run_analysis(data)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # Render will run `python main.py`; expose on 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000)

