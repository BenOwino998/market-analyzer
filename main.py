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

