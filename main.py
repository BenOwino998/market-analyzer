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

# --- Legal pages ---
@app.route("/privacy", methods=["GET"])
def privacy():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="utf-8"><title>Privacy Policy</title></head>
    <body>
      <h1>Privacy Policy</h1>
      <p>This GPT and site are provided for research and educational purposes only.
      No personal data is collected, stored, or shared by the creator of this GPT or site.</p>
      <p>Any data you provide through interactions with the GPT is processed under
      <a href="https://openai.com/policies/privacy-policy" target="_blank" rel="noopener">OpenAI’s Privacy Policy</a>.</p>
    </body>
    </html>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/terms", methods=["GET"])
def terms():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="utf-8"><title>Terms of Use</title></head>
    <body>
      <h1>Terms of Use</h1>
      <p>By using this GPT and site, you agree they are provided “as is” without warranties of any kind.
      The creator is not responsible for how outputs are used.</p>
      <p>You are responsible for ensuring compliance with applicable laws and regulations.
      Use at your own risk.</p>
    </body>
    </html>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}

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

