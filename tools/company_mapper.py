COMPANY_TO_TICKER = {
    "NVIDIA": "NVDA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "TESLA": "TSLA",
    "NETFLIX": "NFLX",

    "JPMORGAN": "JPM",
    "JPMorgan Chase": "JPM",
    "GOLDMAN SACHS": "GS",
    "VISA": "V",
    "MASTERCARD": "MA",
    "PAYPAL": "PYPL",

    "WALMART": "WMT",
    "COSTCO": "COST",
    "COCA COLA": "KO",
    "PEPSI": "PEP",
    "NIKE": "NKE",

    "PFIZER": "PFE",
    "JOHNSON AND JOHNSON": "JNJ",
    "MODERNA": "MRNA",

    "SALESFORCE": "CRM",
    "ADOBE": "ADBE",
    "PALANTIR": "PLTR"
}


def company_to_ticker(company: str):
    return COMPANY_TO_TICKER.get(company.strip().upper())
