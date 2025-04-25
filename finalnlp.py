import streamlit as st
import nltk
import re
import pandas as pd
import seaborn as sns
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob
from transformers import pipeline
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import ssl

# Set up page config
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="▶️",
    layout="wide"
)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# SSL and NLTK setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
@st.cache_resource
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

setup_nltk()

# Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", 
                   model="facebook/bart-large-cnn",
                   tokenizer="facebook/bart-large-cnn",
                   framework="pt")

summarizer = load_summarizer()

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def clean_text(text):
    """Enhanced text cleaning"""
    text = re.sub(r'\[.*?\]', '', text)  # Remove any text in brackets
    text = re.sub(r'\(.*?\)', '', text)  # Remove text in parentheses
    text = re.sub(r'\bhttps?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text

def analyze_keywords(text, top_n=15):
    """Enhanced keyword analysis with lemmatization"""
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text.lower())
    
    # Custom stopwords
    base_stopwords = set(nltk.corpus.stopwords.words('english'))
    extra_stopwords = {'like', 'one', 'would', 'get', 'also', 'could', 'thing', 'really'}
    stopwords = base_stopwords.union(extra_stopwords)
    
    # Process words
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords and len(word) > 2]
    
    return nltk.FreqDist(words).most_common(top_n)

def summarize_text(text):
    """Generate abstractive summary using BART with default length"""
    # Split into manageable chunks (BERT has max input length)
    max_chunk_length = 1024
    text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, 
                            max_length=150,  # Fixed default length
                            min_length=50, 
                            do_sample=False,
                            truncation=True)
        summaries.append(summary[0]['summary_text'])
    
    return ' '.join(summaries)

def analyze_sentiment(text):
    """Perform sentiment analysis on text"""
    blob = TextBlob(text)
    sentences = [{
        'text': str(sent),
        'polarity': sent.sentiment.polarity,
        'subjectivity': sent.sentiment.subjectivity
    } for sent in blob.sentences]
    
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'sentences': sentences
    }

def generate_wordcloud(text):
    """Generate styled word cloud"""
    wordcloud = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='viridis',
        stopwords=set(nltk.corpus.stopwords.words('english'))
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_sentiment(polarities):
    """Enhanced sentiment visualization"""
    plt.figure(figsize=(12, 5))
    sns.set_style("whitegrid")
    
    # Calculate rolling average
    window_size = max(3, len(polarities) // 15)
    rolling_avg = pd.Series(polarities).rolling(window=window_size, min_periods=1).mean()
    
    # Create plot
    ax = sns.lineplot(x=range(len(polarities)), y=polarities, 
                     alpha=0.3, label='Instant Polarity')
    sns.lineplot(x=range(len(rolling_avg)), y=rolling_avg, 
                color='#FF6B6B', label=f'Trend Line ({window_size} sentences)')
    
    # Add fill colors
    ax.fill_between(range(len(polarities)), polarities, 
                   where=(pd.Series(polarities) >= 0), 
                   color='#4ECDC4', alpha=0.1, label='Positive Area')
    ax.fill_between(range(len(polarities)), polarities, 
                   where=(pd.Series(polarities) < 0), 
                   color='#FF6B6B', alpha=0.1, label='Negative Area')
    
    plt.title("Sentiment Analysis Over Video Duration", fontsize=14, pad=15)
    plt.xlabel("Sentence Position in Transcript", labelpad=12)
    plt.ylabel("Polarity Score", labelpad=12)
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_keywords(keywords):
    """Modern keyword visualization"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    words = [w[0] for w in keywords][::-1]
    freqs = [w[1] for w in keywords][::-1]
    
    # Create gradient colors
    colors = sns.color_palette("magma", len(words))
    
    bars = plt.barh(words, freqs, color=colors, height=0.7)
    plt.bar_label(bars, padding=3, labels=[f"{f:,}" for f in freqs], fontsize=8)
    
    plt.title("Top Keywords Frequency", fontsize=14, pad=15)
    plt.xlabel("Frequency Count", labelpad=12)
    plt.ylabel("Keywords", labelpad=12)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# Streamlit UI
st.title("🎬 YouTube Video Analyzer")
st.markdown("---")

# Input section
url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Video") and url:
    with st.spinner("Processing video..."):
        try:
            video_id = extract_video_id(url)
            if not video_id:
                st.error("❌ Invalid YouTube URL")
                st.stop()

            # Get transcript with proper punctuation handling
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Build text with proper sentence segmentation
            video_text = []
            current_sentence = []
            
            for entry in transcript:
                text = entry['text'].strip()
                if not text:
                    continue
                
                # Add to current sentence
                current_sentence.append(text)
                
                # Check for sentence end
                if text[-1] in '.!?':
                    joined_sentence = ' '.join(current_sentence)
                    video_text.append(joined_sentence)
                    current_sentence = []
            
            # Add any remaining text
            if current_sentence:
                video_text.append(' '.join(current_sentence))
            
            full_text = ' '.join(video_text)
            cleaned_text = clean_text(full_text)
            
            # Create analysis tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "📊 Sentiment", "🔑 Keywords", "📈 Full Data"])
            
            with tab1:
                st.subheader("Video Summary")
                with st.spinner("Generating AI-powered summary..."):
                    summary = summarize_text(cleaned_text)
                    st.markdown(f"**AI Summary ({len(summary.split())} words):**")
                    st.write(summary)
            
            with tab2:
                st.subheader("Sentiment Analysis")
                sentiment = analyze_sentiment(cleaned_text)
                
                # Metrics
                col1, col2 = st.columns(2)
                col1.metric("Overall Polarity", 
                          f"{sentiment['polarity']:.2f}", 
                          "Positive" if sentiment['polarity'] > 0 else "Negative")
                col2.metric("Content Subjectivity", 
                          f"{sentiment['subjectivity']:.2f}", 
                          "Subjective" if sentiment['subjectivity'] > 0.5 else "Objective")
                
                # Sentiment visualization
                plot_sentiment([s['polarity'] for s in sentiment['sentences']])
                
                # Key sentences
                st.subheader("Key Insights")
                col1, col2 = st.columns(2)
                most_positive = max(sentiment['sentences'], key=lambda x: x['polarity'])
                most_negative = min(sentiment['sentences'], key=lambda x: x['polarity'])
                
                with col1:
                    st.markdown("😊 **Most Positive Moment**")
                    st.success(f'"{most_positive["text"]}" \n\n(Polarity: {most_positive["polarity"]:.2f})')
                
                with col2:
                    st.markdown("😞 **Most Critical Moment**")
                    st.error(f'"{most_negative["text"]}" \n\n(Polarity: {most_negative["polarity"]:.2f})')
            
            with tab3:
                st.subheader("Keyword Analysis")
                st.markdown("### ☁️ Conceptual Word Cloud")
                generate_wordcloud(cleaned_text)
                
                st.markdown("### 📌 Top Keywords Frequency")
                keywords = analyze_keywords(cleaned_text)
                plot_keywords(keywords)
            
            with tab4:
                st.subheader("Complete Analysis Data")
                col1, col2 = st.columns(2)
                col1.metric("Total Words", len(cleaned_text.split()))
               
                st.markdown("### 📄 Full Transcript Sentences")
                for i, sentence in enumerate(sentiment['sentences']):
                    with st.expander(f"Sentence {i+1} (Polarity: {sentence['polarity']:.2f})"):
                        st.write(sentence['text'])
            
            st.session_state.analyzed = True

        except Exception as e:
            error_message = str(e)
            st.error(f"Analysis Error: {error_message}")
            if "TranscriptDisabled" in error_message:
                st.info("ℹ️ This video has disabled subtitles")
            elif "NoTranscriptFound" in error_message:
                st.info("ℹ️ No transcript available for this video")

st.markdown("---")