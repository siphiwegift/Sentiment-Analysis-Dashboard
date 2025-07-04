import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import re
from fpdf import FPDF

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

@st.cache_resource
def load_sentiment_model():
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}. Please check your internet connection.")
        return None

sentiment_analyzer = load_sentiment_model()
if sentiment_analyzer is None:
    st.stop()

# --- Helper Functions ---
def analyze_sentiment(text):
    if not text or not text.strip():
        return {"label": "Neutral", "score": 1.0, "explanation": "Empty text treated as neutral."}
    try:
        result = sentiment_analyzer(text)[0]
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        result['label'] = label_map.get(result['label'], result['label'])
        return result
    except Exception as e:
        st.error(f"Error analyzing text: '{text[:50]}...'. Error: {e}")
        return {"label": "Error", "score": 0.0, "explanation": f"Failed: {e}"}

def extract_keywords_simple(text, sentiment_label):
    if not text or not text.strip():
        return []
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = set()
    positive_words = ["good", "great", "excellent", "amazing", "love", "happy", "fantastic", "perfect", "awesome", "best", "enjoy", "pleased", "recommend", "wonderful", "brilliant", "superb"]
    negative_words = ["bad", "terrible", "poor", "awful", "hate", "unhappy", "disappointing", "worst", "frustrating", "problem", "issue", "slow", "bug", "horrible", "difficult", "fail"]
    neutral_filler_words = ["is", "the", "a", "an", "it", "this", "that", "and", "but", "or", "so", "very", "much", "little", "some", "many", "few", "can", "will", "would", "in", "on", "at", "for", "with", "to", "of", "from", "by"]
    
    # Filter out common filler words from the input text
    relevant_words = [word for word in words if word not in neutral_filler_words]
    
    # Add words from the text that match our predefined sentiment keywords
    for word in relevant_words:
        if sentiment_label == "Positive" and word in positive_words:
            keywords.add(word)
        elif sentiment_label == "Negative" and word in negative_words:
            keywords.add(word)
    
    # Simple addition of highly relevant words from the text itself, regardless of predefined lists
    # This ensures keywords directly from the text are captured even if not in the fixed lists
    # We can refine this with more sophisticated NLP techniques (e.g., TF-IDF, POS tagging) for better keyword extraction
    for word in relevant_words:
        if word not in keywords and len(word) > 2: # Avoid very short words as general keywords
            # A very basic heuristic: if a word appears and is not a filler, consider it a potential keyword.
            # This part could be significantly improved for accuracy.
            keywords.add(word)

    return list(keywords)


def highlight_text_with_keywords(text, keywords, sentiment_label):
    color_map = {
        "Positive": "background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 3px;",
        "Negative": "background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 3px;",
        "Neutral": "background-color: #e2e3e5; color: #383d41; padding: 2px 4px; border-radius: 3px;",
        "Error": "background-color: #fff3cd; color: #856404; padding: 2px 4px; border-radius: 3px;"
    }
    style = color_map.get(sentiment_label, "")
    # Sort keywords by length in descending order to prevent partial matches of shorter keywords
    # within longer ones (e.g., "bad" within "badly").
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    highlighted_text = text
    for keyword in sorted_keywords:
        # Use regex to find whole word matches and replace them with highlighted spans
        highlighted_text = re.sub(r'\b(' + re.escape(keyword) + r')\b', f'<span style="{style}">\\1</span>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(0, 10, "Sentiment Analysis Report", 0, 1, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 7, body)
        self.ln()

def generate_pdf(df):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title("Analysis Summary")
    
    total = len(df)
    pos = (df["Sentiment"] == "Positive").sum()
    neu = (df["Sentiment"] == "Neutral").sum()
    neg = (df["Sentiment"] == "Negative").sum()

    summary_text = (
        f"Total Texts Analyzed: {total}\n"
        f"Positive Sentiments: {pos} ({pos / total:.1%})\n"
        f"Neutral Sentiments: {neu} ({neu / total:.1%})\n"
        f"Negative Sentiments: {neg} ({neg / total:.1%})\n"
    )
    pdf.chapter_body(summary_text)

    if not df.empty:
        pdf.chapter_title("Detailed Results")
        pdf.set_font("Arial", 'B', 8)
        # Define column widths, adjust as necessary to fit content
        col_widths = [70, 25, 25, 60] 
        headers = ["Original Text", "Sentiment", "Confidence", "Keywords"]
        
        # Print header row
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, border=1, ln=False, align='C')
        pdf.ln()
        
        pdf.set_font("Arial", size=7) # Smaller font for detailed rows
        
        # Iterate over DataFrame rows to add to PDF
        for _, row in df.iterrows():
            # Check if current position is too close to bottom of page, if so, add new page
            if pdf.get_y() > 250: # Arbitrary threshold, adjust based on footer size
                pdf.add_page()
                pdf.set_font("Arial", 'B', 8) # Re-apply header font
                for i, header in enumerate(headers):
                    pdf.cell(col_widths[i], 7, header, border=1, ln=False, align='C')
                pdf.ln()
                pdf.set_font("Arial", size=7) # Re-apply row font

            text_cell = row["Original Text"]
            if len(text_cell) > 60: # Truncate long text for PDF display
                text_cell = text_cell[:57] + "..."
            
            keywords_cell = str(row["Keywords"])
            if len(keywords_cell) > 50: # Truncate long keywords for PDF display
                keywords_cell = keywords_cell[:47] + "..."

            # Use multi_cell for text to allow wrapping if content is too long for a single cell
            # This is a simplification; for proper wrapping in tables, more complex logic is needed
            # For simplicity here, we're still using cell and truncating.
            pdf.cell(col_widths[0], 7, text_cell, border=1)
            pdf.cell(col_widths[1], 7, row["Sentiment"], border=1, align='C')
            pdf.cell(col_widths[2], 7, str(row["Confidence Score"]), border=1, align='C')
            pdf.cell(col_widths[3], 7, keywords_cell, border=1)
            pdf.ln()
            
    return pdf.output(dest='S').encode('latin1')

# --- UI ---
st.title("🗣️ Sentiment Analysis Dashboard")
st.markdown("""
This dashboard helps you analyze the emotional tone of text, whether it's customer reviews, social media posts, or any other textual data.
""")

input_method = st.radio("Choose input method:", ("Direct Text Entry", "File Upload (CSV/TXT)"), horizontal=True)
texts_to_analyze = []

if input_method == "Direct Text Entry":
    user_input = st.text_area("Enter text here (one entry per line for multiple analyses):", height=150, 
                               placeholder="e.g., 'This product is amazing!' or 'Service was terrible.'")
    if user_input:
        # Split by new line for multiple entries
        texts_to_analyze = [line.strip() for line in user_input.split('\n') if line.strip()]
else:
    uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])
    if uploaded_file:
        ext = uploaded_file.name.split('.')[-1].lower()
        try:
            if ext == "csv":
                df = pd.read_csv(uploaded_file)
                # Try to find a 'text' column, otherwise use the first column
                if 'text' in df.columns:
                    texts_to_analyze = df['text'].astype(str).tolist()
                elif not df.empty:
                    st.warning("No 'text' column found. Using the first column for analysis.")
                    texts_to_analyze = df.iloc[:, 0].astype(str).tolist()
                else:
                    st.warning("CSV file is empty or could not be read.")
            elif ext == "txt":
                texts_to_analyze = uploaded_file.read().decode("utf-8").splitlines()
                texts_to_analyze = [line.strip() for line in texts_to_analyze if line.strip()]
            
            if not texts_to_analyze:
                st.info("No valid text entries found in the uploaded file.")

        except Exception as e:
            st.error(f"Error reading file: {e}. Please ensure it's a valid CSV/TXT format with readable text.")

# --- Analysis ---
if texts_to_analyze:
    st.subheader("Analysis Results")
    # Initialize Streamlit components for progress feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    for i, text in enumerate(texts_to_analyze):
        # Update progress and status
        progress_bar.progress((i + 1) / len(texts_to_analyze))
        status_text.text(f"Analyzing text {i+1}/{len(texts_to_analyze)}...")
        
        # Perform sentiment analysis and keyword extraction
        result = analyze_sentiment(text)
        keywords = extract_keywords_simple(text, result['label'])
        
        # Store results
        results.append({
            "Original Text": text,
            "Sentiment": result['label'],
            "Confidence Score": f"{result['score']:.2f}",
            "Keywords": ", ".join(keywords) if keywords else "N/A",
            "Highlighted Text": highlight_text_with_keywords(text, keywords, result['label']),
            "Explanation": result.get("explanation", f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
        })
    
    # Finalize progress feedback
    status_text.text("Analysis complete!")
    progress_bar.empty() # Remove the progress bar

    df_results = pd.DataFrame(results)
    st.dataframe(df_results[['Original Text', 'Sentiment', 'Confidence Score', 'Keywords']], use_container_width=True)

    # --- Batch Summary ---
    st.markdown("---")
    st.markdown("### 📊 Batch Summary")
    
    total = len(df_results)
    if total > 0: # Ensure there are results before calculating percentages
        pos = (df_results["Sentiment"] == "Positive").sum()
        neu = (df_results["Sentiment"] == "Neutral").sum()
        neg = (df_results["Sentiment"] == "Negative").sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Texts", total)
        c2.metric("Positive", pos)
        c3.metric("Neutral", neu)
        c4.metric("Negative", neg)

        st.markdown(f"""
        - **Positive:** {pos} ({pos / total:.1%})  
        - **Neutral:** {neu} ({neu / total:.1%})  
        - **Negative:** {neg} ({neg / total:.1%})
        """)

        st.markdown("#### 🔍 Confidence Statistics")
        # Convert confidence scores to float for numerical operations
        conf_vals = df_results['Confidence Score'].astype(float)
        st.write(f"- **Average Confidence:** {conf_vals.mean():.2f}")
        st.write(f"- **Highest Confidence:** {conf_vals.max():.2f} | **Lowest Confidence:** {conf_vals.min():.2f}")

        # --- Charts ---
        st.markdown("---")
        st.markdown("### 📈 Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_results, x="Sentiment", palette="viridis", ax=ax, order=["Positive", "Neutral", "Negative", "Error"])
        ax.set_title("Distribution of Sentiments")
        ax.set_xlabel("Sentiment Category")
        ax.set_ylabel("Number of Texts")
        st.pyplot(fig)

        if len(df_results) > 1: # Only show confidence distribution if more than one text
            st.markdown("---")
            st.markdown("### 📈 Confidence Score Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(conf_vals, kde=True, bins=10, ax=ax2, color='skyblue')
            ax2.set_title("Distribution of Confidence Scores")
            ax2.set_xlabel("Confidence Score")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

    else:
        st.info("No valid sentiments to display summaries or charts.")


    # --- Paginated Viewer ---
    st.markdown("---")
    st.markdown("### 🔎 Browse Analyzed Texts with Highlights")
    
    # Allow user to set page size
    col_page_size, col_spacer = st.columns([1, 3])
    with col_page_size:
        page_size = st.slider("Items per page:", 1, 20, 5)
    
    total_texts = len(df_results)
    pages = (total_texts - 1) // page_size + 1
    
    # Handle cases where there are no results to prevent division by zero or errors
    if total_texts == 0:
        st.info("No texts to browse.")
    else:
        page = st.number_input(f"Page (1-{pages}):", 1, pages, 1)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_texts)

        for idx in range(start_idx, end_idx):
            r = df_results.iloc[idx]
            st.markdown(f"#### Text {idx+1}")
            st.write(f"**Sentiment:** {r['Sentiment']} | **Confidence:** {r['Confidence Score']}")
            st.write(f"**Keywords:** {r['Keywords']}")
            st.markdown(f"**Original Text with Highlights:**")
            st.markdown(r["Highlighted Text"], unsafe_allow_html=True)
            st.markdown(f"**Explanation:** {r['Explanation']}")
            st.markdown("---")

    # --- Export ---
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )
    with col2:
        json_buffer = io.StringIO()
        df_results.to_json(json_buffer, orient="records", indent=2)
        st.download_button(
            label="Download JSON",
            data=json_buffer.getvalue(),
            file_name="sentiment_analysis_results.json",
            mime="application/json"
        )
    with col3:
        try:
            pdf_bytes = generate_pdf(df_results)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

else:
    st.info("Enter text directly or upload a CSV/TXT file to begin sentiment analysis.")