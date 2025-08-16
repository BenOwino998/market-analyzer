from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class MarketQuery(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.post("/analyze")
def analyze_stock(query: MarketQuery):
    return {
        "message": f"Received request to analyze {query.ticker} from {query.start_date} to {query.end_date}."
    }
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    message = data.get("message", "")
    return jsonify({"response": f"Received message: {message}"})

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Fractal Market Adapter is running."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

