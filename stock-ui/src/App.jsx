import { useState } from "react";
import axios from "axios";
import "./App.css";

/* 👇 Company list shown to user */
const popularCompanies = [
  "Adobe",
  "Amazon",
  "Apple",
  "Costco",
  "Coca Cola",
  "Google",
  "Goldman Sachs",
  "Johnson and Johnson",
  "JPMorgan",
  "JPMorgan Chase",
  "Mastercard",
  "Meta",
  "Microsoft",
  "Moderna",
  "Netflix",
  "Nike",
  "NVIDIA",
  "Palantir",
  "PayPal",
  "Pepsi",
  "Pfizer",
  "Salesforce",
  "Tesla",
  "Visa",
  "Walmart"
];

export default function App() {
  const [company, setCompany] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  /* 🔥 NORMALIZED API CALL */
  const analyzeStock = async () => {
    if (!company.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const normalizedCompany = company.trim().toUpperCase();

      const res = await axios.post("http://127.0.0.1:8000/analyze", {
        company_name: normalizedCompany,
      });

      setResult(res.data);
    } catch (err) {
      setError("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const displayValue = (value) =>
    value === null || value === undefined || value === ""
      ? "Not provided"
      : value;

  return (
    <div className="app-container">
      <div className="header">
        <h1>📊 AI Stock Market Analyzer</h1>
        <p>
          Multi-agent intelligence combining macro, technical, and fundamental
          analysis
        </p>
      </div>

      {/* INPUT + BUTTON */}
      <div className="input-row">
        <input
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          placeholder="Enter company name (Apple, NVIDIA, Tesla...)"
        />
        <button onClick={analyzeStock} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {/* 👇 COMPANY QUICK SELECT */}
      <div className="company-list">
        <span className="company-list-label">Popular companies:</span>
        {popularCompanies.map((name) => (
          <button
            key={name}
            className="company-chip"
            onClick={() => {
              setCompany(name);
              setError("");
            }}
          >
            {name}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card">
          <div className="card-header">
            <h2>
              {result.company} <span>({result.ticker})</span>
            </h2>
            <span className={`badge ${result.action?.toLowerCase()}`}>
              {result.action}
            </span>
          </div>

          <div className="metrics">
            <div className="metric">
              <span className="metric-label">Confidence</span>
              <span className="metric-value">
                {displayValue(result.confidence)}%
              </span>
            </div>

            <div className="metric">
              <span className="metric-label">Expected ROI</span>
              <span className="metric-value">
                {displayValue(result.expected_roi)}%
              </span>
            </div>

            <div className="metric">
              <span className="metric-label">Horizon</span>
              <span className="metric-value">
                {displayValue(result.horizon)}
              </span>
            </div>

            <div className="metric">
              <span className="metric-label">Entry</span>
              <span className="metric-value">
                {displayValue(result.entry)}
              </span>
            </div>

            <div className="metric">
              <span className="metric-label">Target</span>
              <span className="metric-value">
                {displayValue(result.target)}
              </span>
            </div>

            <div className="metric">
              <span className="metric-label">Stop Loss</span>
              <span className="metric-value">
                {displayValue(result.stop_loss)}
              </span>
            </div>
          </div>

          <div className="reasoning">
            <h3>🧠 AI Reasoning</h3>
            <p>{result.reason}</p>
          </div>
        </div>
      )}
    </div>
  );
}
