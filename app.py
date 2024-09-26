import streamlit as st
import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('universal_tagset')
# Download the averaged_perceptron_tagger (without _eng suffix)
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')  # Download the universal tagset as well

# Streamlit app
st.title("Text Analysis Workshop")

# Large text input field
st.header("Input Your Text")
user_text = st.text_area("Enter your text below:", height=300)

if user_text:
    st.header("Explore Text Metrics")

    # Tokenize text
    tokens = word_tokenize(user_text.lower())
    sentences = sent_tokenize(user_text)

    # Word and Character Count
    if st.button("Word & Character Count"):
        num_words = len(tokens)  # Ensure num_words is defined here
        num_chars = len(user_text)
        avg_word_length = sum(len(word) for word in tokens) / num_words

        st.write(f"Total Words: {num_words}")
        st.write(f"Total Characters: {num_chars}")
        st.write(f"Average Word Length: {avg_word_length:.2f} characters")

        # Find the 5 longest words
        longest_words = sorted(tokens, key=len, reverse=True)[:5]

        # Display the 5 longest words
        st.write("5 Longest Words:")
        for word in longest_words:
            st.write(f"{word} ({len(word)} characters)")

    # Sentence Count and Length
    if st.button("Sentence Analysis"):
        num_sentences = len(sentences)
        num_words = len(tokens)  # Ensure num_words is calculated again
        avg_sentence_length = num_words / num_sentences
        longest_sentences = sorted(sentences, key=len, reverse=True)[:2]

        st.write(f"Total Sentences: {num_sentences}")
        st.write(f"Average Sentence Length: {avg_sentence_length:.2f} words")
        st.write("Longest Sentences:")
        st.write(longest_sentences[0])
        st.write(longest_sentences[1])

        # Plot sentence length through time
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(sentence_lengths)), sentence_lengths, marker='o')
        plt.xlabel('Sentence Index')
        plt.ylabel('Sentence Length (words)')
        plt.title('Sentence Length Over Time')
        st.pyplot(plt)

    # Type-Token Ratio (Lexical Diversity)
    if st.button("Type-Token Ratio"):
        types = set(tokens)
        ttr = len(types) / len(tokens)
        st.write(f"Type-Token Ratio (TTR): {ttr:.2f}")

    # Word Frequency Distribution
    if st.button("Word Frequency Distribution"):
        freq_dist = FreqDist(tokens)
        most_common_words = freq_dist.most_common(10)
        
        # Display the top 10 most frequent words as a simple text list
        st.write("Top 10 Most Frequent Words:")
        for word, count in most_common_words:
            st.write(f"{word}: {count} occurrences")

        # Plot word frequency
        words, counts = zip(*most_common_words)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(words), y=list(counts))
        plt.title("Word Frequency Distribution")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    # Sentiment Analysis
    if st.button("Sentiment Analysis"):
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = [sid.polarity_scores(sentence)['compound'] for sentence in sentences]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
        if avg_sentiment > 0:
            st.write("Overall Sentiment: Positive")
        elif avg_sentiment < 0:
            st.write("Overall Sentiment: Negative")
        else:
            st.write("Overall Sentiment: Neutral")

        # Sentiment Visualization
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(sentiment_scores)), sentiment_scores, color='green' if avg_sentiment > 0 else 'red')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("Sentence Index")
        plt.ylabel("Sentiment Score")
        plt.title("Sentiment Score per Sentence")
        st.pyplot(plt)

    # Part-of-Speech (POS) Tagging
    if st.button("POS Tagging"):
        pos_tags = nltk.pos_tag(tokens, tagset='universal')
        pos_counts = nltk.FreqDist(tag for word, tag in pos_tags)

        st.write("POS Tag Distribution:")
        st.write(dict(pos_counts))

        # Plot POS distribution
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()))
        plt.title("Part-of-Speech (POS) Distribution")
        plt.xlabel("POS Tag")
        plt.ylabel("Frequency")
        st.pyplot(plt)
