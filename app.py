import streamlit as st
import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import seaborn as sns
import re
from streamlit.components.v1 import html

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Default text for the text area
default_text = """

"""

# Clear figures before plotting to avoid overlapping
def plot_sentence_lengths(sentences):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentence_lengths)), sentence_lengths, marker='o')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Length (words)')
    plt.title('Sentence Length Over Time')
    st.pyplot(plt)


def plot_ttr_over_time(tokens, window_size):
    ttr_values = []
    for i in range(0, len(tokens) - window_size + 1, window_size):
        window_tokens = tokens[i:i + window_size]
        types = set(window_tokens)
        ttr = len(types) / len(window_tokens)
        ttr_values.append(ttr)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(ttr_values)), ttr_values, marker='o', linestyle='-', color='purple')
    plt.xlabel("Window Index")
    plt.ylabel("Type-Token Ratio (TTR)")
    plt.title("Type-Token Ratio Over Time")
    st.pyplot(plt)


def plot_word_frequency(most_common_words):
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title("Word Frequency Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)


def plot_sentiment(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['green' if score > 0 else 'red' if score < 0 else 'yellow' for score in sentiment_scores])
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score per Sentence")
    st.pyplot(plt)


def plot_sentiment_line(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentiment_scores)), sentiment_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score Zigzag Line")
    st.pyplot(plt)


# Function to highlight longest words in the text
def highlight_longest_words(text, longest_words):
    for word in longest_words:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(pattern, f'<span style="color: red; font-weight: bold;">{word}</span>', text, flags=re.IGNORECASE)
    return text


# Streamlit app
st.title("Text Analysis Workshop")

# Large text input field
st.header("Input Your Text")
user_text = st.text_area("Enter your text below:", value=default_text, height=300)

# Slider for adjustable parameters
st.sidebar.header("Adjustable Parameters")
window_size = st.sidebar.slider("Window Size for TTR Analysis", min_value=10, max_value=100, value=50, step=10)

if user_text:
    # Tokenize text
    try:
        tokens = word_tokenize(user_text.lower())
        sentences = sent_tokenize(user_text)
    except LookupError:
        st.error("Required NLTK resources are missing. Please ensure 'punkt' is downloaded.")
        tokens, sentences = [], []

    # Create tabs for each analysis feature
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Word & Character Count", "Sentence Analysis", "Type-Token Ratio", "Word Frequency Distribution", "Sentiment Analysis"])

    with tab1:
        st.header("Word & Character Count")
        num_words = len(tokens)
        num_chars = len(user_text)
        avg_word_length = sum(len(word) for word in tokens) / num_words if num_words > 0 else 0
        st.write(f"Total Words: {num_words}")
        st.write(f"Total Characters: {num_chars}")
        st.write(f"Average Word Length: {avg_word_length:.2f} characters")
        longest_words = sorted(tokens, key=len, reverse=True)[:5]
        st.write("5 Longest Words:")
        for word in longest_words:
            st.write(f"{word} ({len(word)} characters)")

        # Highlight longest words in the text
        highlighted_text = highlight_longest_words(user_text, longest_words)
        st.markdown("### Highlighted Text with Longest Words")
        html_code = f'<div style="white-space: pre-wrap;">{highlighted_text}</div>'
        html(html_code, height=300)

    with tab2:
        st.header("Sentence Analysis")
        num_sentences = len(sentences)
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        st.write(f"Total Sentences: {num_sentences}")
        st.write(f"Average Sentence Length: {avg_sentence_length:.2f} words")
        if num_sentences > 0:
            longest_sentences = sorted(sentences, key=len, reverse=True)[:2]
            st.write("Longest Sentences:")
            st.write(longest_sentences[0])
            if len(longest_sentences) > 1:
                st.write(longest_sentences[1])
            plot_sentence_lengths(sentences)

    with tab3:
        st.header("Type-Token Ratio")
        types = set(tokens)
        ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
        st.write(f"Type-Token Ratio (TTR): {ttr:.2f}")

        # Plot TTR over time with adjustable window size
        if len(tokens) >= window_size:
            st.write("Type-Token Ratio Over Time (using a sliding window):")
            plot_ttr_over_time(tokens, window_size)

        # Display lexical richness metrics
        hapax_legomena = [word for word in tokens if tokens.count(word) == 1]
        st.write(f"Number of Hapax Legomena (words that occur only once): {len(hapax_legomena)}")
        st.write(f"Percentage of Hapax Legomena: {(len(hapax_legomena) / len(tokens)) * 100:.2f}%" if len(tokens) > 0 else "Percentage of Hapax Legomena: 0%")

    with tab4:
        st.header("Word Frequency Distribution")
        if tokens:
            freq_dist = FreqDist(tokens)
            most_common_words = freq_dist.most_common(10)
            st.write("Top 10 Most Frequent Words:")
            for word, count in most_common_words:
                st.write(f"{word}: {count} occurrences")
            plot_word_frequency(most_common_words)

    with tab5:
        st.header("Sentiment Analysis")
        if sentences:
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if len(sentiment_scores) > 0 else 0
            st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
            if avg_sentiment > 0:
                st.write("Overall Sentiment: Positive")
            elif avg_sentiment < 0:
                st.write("Overall Sentiment: Negative")
            else:
                st.write("Overall Sentiment: Neutral")
            plot_sentiment(sentiment_scores)
            plot_sentiment_line(sentiment_scores)

# Option to download results
if user_text:
    st.sidebar.header("Download Analysis Report")
    if st.sidebar.button("Generate Report"):
        st.write("Feature not implemented yet. Placeholder for future download feature.")

