import axios from "axios";

const API_URL = "http://127.0.0.1:8000/analyze";

export async function analyzeCompany(companyName) {
    const response = await axios.post(API_URL, {
        company_name: companyName,
    });
    return response.data;
}
