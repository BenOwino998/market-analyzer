from flask import Flask, jsonify, request, send_from_directory
from FAA_MVP import run_analysis  # entry point from your FAA module

app = Flask(__name__)

# Serve plugin files
@app.route("/.well-known/<path:filename>", methods=["GET"])
def well_known(filename):
    return send_from_directory(".well-known", filename)

# Health check
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello from Fractal Market Analyzer (Flask)!"})

# Main analysis endpoint
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True, silent=False)
        result = run_analysis(data)  # call FAA
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # Render runs `python main.py`; bind to 0.0.0.0 for external access
    app.run(host="0.0.0.0", port=5000)

