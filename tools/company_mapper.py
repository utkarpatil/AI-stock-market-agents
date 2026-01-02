COMPANY_TO_TICKER = {
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "NETFLIX": "NFLX",
    "META": "META",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL"
}

def company_to_ticker(company: str):
    return COMPANY_TO_TICKER.get(company.strip().upper())
