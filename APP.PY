import streamlit as st
from transformers import pipeline

# Load models (with caching)
@st.cache_resource
def load_models():
    # Force both models to use CPU
    summarizer = pipeline("summarization", device=-1)
    sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
    return summarizer, sentiment_analyzer

summarizer, sentiment_analyzer = load_models()

# Page setup
st.set_page_config(page_title="Text Analysis App", layout="centered")
st.title("🧠 Text Summarization + Sentiment Analysis")

# Input
text_input = st.text_area("📝 Enter text to analyze", height=300, placeholder="Paste your paragraph, article, or review...")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Summarization Settings")
    max_len = st.slider("Maximum Summary Length", 50, 300, 130, 10)
    min_len = st.slider("Minimum Summary Length", 10, 100, 30, 5)
    st.markdown("---")
    st.info("This app summarizes text and detects overall sentiment.")

# Analyze button
if st.button("🚀 Analyze Text"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):

            # Summarization
            summary = summarizer(text_input, max_length=max_len, min_length=min_len, do_sample=False)
            summary_text = summary[0]['summary_text']

            # Sentiment analysis
            sentiment = sentiment_analyzer(text_input)
            label = sentiment[0]['label']
            score = sentiment[0]['score']

        # Results
        st.success("✅ Analysis Complete!")

        st.subheader("🧾 Summary")
        st.markdown(f"> {summary_text}")

        st.subheader("📊 Sentiment Analysis")
        emoji = "😊" if label == "POSITIVE" else "😐" if label == "NEUTRAL" else "😞"
        st.markdown(f"**Sentiment**: {label} {emoji}  \n**Confidence**: {score:.2f}")

        # Optionally: allow download
        st.download_button("⬇️ Download Summary", summary_text, file_name="summary.txt")

