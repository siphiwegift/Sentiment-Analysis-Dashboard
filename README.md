📊 Sentiment Analysis Dashboard
This is a simple yet powerful Sentiment Analysis App built with Streamlit and KeyBERT. It classifies text feedback as Positive, Negative, or Neutral, extracts keywords, shows a word cloud, and lets you export results as CSV, JSON, and PDF.

🚀 Features
✅ Rule-based sentiment detection using positive/negative keywords
✅ Confidence scoring for sentiment classification
✅ Automatic keyword extraction using KeyBERT
✅ Visualizations: Sentiment distribution bar chart, word cloud
✅ Export results: CSV, JSON, PDF report
✅ Easy to use with direct text entry or file upload

⚙️ Tech Stack
Python 3.11
Streamlit
KeyBERT
Sentence-Transformers
FPDF
Matplotlib
WordCloud
📥 Installation
1️⃣ Clone this repository:

git clone https://github.com/your-username/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
2️⃣ Create virtual environment (recommended):

python -m venv venv
# Activate:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3️⃣ Install dependencies:

pip install -r requirements.txt
🏃 Run the app
streamlit run PythonApp.py
✏️ How to use
1️⃣ Select Input Mode in the sidebar
2️⃣ Click Analyze to see sentiment, keywords, charts, and word cloud
3️⃣ Download your results in CSV, JSON, PDF formats

🔍 API Selection Justification
KeyBERT for local keyword extraction (no API key needed).
Simple keyword-based sentiment rules for clarity.
✅ License
MIT License
