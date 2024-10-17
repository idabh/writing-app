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


# Default text for the text area
default_text = """
It was a queer, sultry summer, the summer they electrocuted the Rosenbergs, and I didn't know what I was doing in New York. I'm stupid about executions.
The idea of being electrocuted makes me sick, and that's all there was to read about in the papers — goggle-eyed headlines staring up at me on every street corner and at the fusty, peanut-smelling
mouth of every subway.
It had nothing to do with me, but I couldn't help wondering what it would be like, being burned alive all along your nerves.

I thought it must be the worst thing in the world.

New York was bad enough.
By nine in the morning the fake, country-wet freshness that somehow seeped in overnight evaporated like the tail end of a sweet dream. Mirage-grey at the bottom of their granite canyons, the hot streets
 wavered in the sun, the car tops sizzled and glittered, and the dry, tindery dust blew into my eyes and down my throat.

I kept hearing about the Rosenbergs over the radio and at the office till I couldn't get them out of my mind.
It was like the first time I saw a cadaver.
For weeks afterwards, the cadaver's head—or what there was left of it—floated up behind my eggs and bacon at breakfast and behind the face of Buddy Willard, who was responsible for my seeing it
in the first place, and pretty soon I felt as though I were carrying that cadaver's head around with me on a string, like some black, noseless balloon stinking of vinegar.

I knew something was wrong with me that summer, because all I could think about was the Rosenbergs and how stupid I'd been to buy all those uncomfortable, expensive clothes,
hanging limp as fish in my closet, and how all the little successes I'd totted up so happily at college fizzled to nothing outside the slick marble and plate-glass fronts along Madison Avenue.

I was supposed to be having the time of my life.

I was supposed to be the envy of thousands of other college girls just like me all over America who wanted nothing more than to be tripping about in those same size seven patent leather shoes I'd bought
in Bloomingdale's one lunch hour with a black patent leather belt and black patent leather pocket-book to match. And when my picture came out in the magazine the twelve of us were working
on — drinking martinis in a skimpy, imitation silver-lamé bodice stuck on to a big, fat cloud of white tulle, on some Starlight Roof, in the company of several anonymous young men
with all-American bone structures hired or loaned for the occasion — everybody would think I must be having a real whirl.

Look what can happen in this country, they'd say.
A girl lives in some out-of-the-way town for nineteen years, so poor she can't afford a magazine, and then she gets a scholarship to college and wins a prize here and a prize there and ends up steering New York like her own private car.

Only I wasn't steering anything, not even myself.
I just bumped from my hotel to work and to parties and from parties to my hotel and back to work like a numb trolley-bus.
I guess I should have been excited the way most of the other girls were, but I couldn't get myself to react.
I felt very still and very empty, the way the eye of a tornado must feel, moving dully along in the middle of the surrounding hullabaloo.
"""

# Clear figures before plotting to avoid overlapping
def plot_sentence_lengths(sentences):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sentence_lengths)), sentence_lengths, marker='o')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentence Length (words)')
    plt.title('Sentence Length Over Time')
    st.pyplot(plt)

def plot_word_frequency(most_common_words):
    words, counts = zip(*most_common_words)
    plt.clf()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title("Word Frequency Distribution")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)

def plot_sentiment(sentiment_scores, avg_sentiment):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['green' if score > 0 else 'red' if score < 0 else 'yellow' for score in sentiment_scores])
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score per Sentence")
    st.pyplot(plt)

# Streamlit app
st.title("Text Analysis Workshop")

# Large text input field
st.header("Input Your Text")
user_text = st.text_area("Enter your text below:", height=300)

if user_text:
    # Tokenize text
    tokens = word_tokenize(user_text.lower())
    sentences = sent_tokenize(user_text)

    # Create tabs for each analysis feature
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Word & Character Count", "Sentence Analysis", "Type-Token Ratio", "Word Frequency Distribution", "Sentiment Analysis"])

    with tab1:
        st.header("Word & Character Count")
        num_words = len(tokens)
        num_chars = len(user_text)
        avg_word_length = sum(len(word) for word in tokens) / num_words
        st.write(f"Total Words: {num_words}")
        st.write(f"Total Characters: {num_chars}")
        st.write(f"Average Word Length: {avg_word_length:.2f} characters")
        longest_words = sorted(tokens, key=len, reverse=True)[:5]
        st.write("5 Longest Words:")
        for word in longest_words:
            st.write(f"{word} ({len(word)} characters)")

    with tab2:
        st.header("Sentence Analysis")
        num_sentences = len(sentences)
        avg_sentence_length = num_words / num_sentences
        st.write(f"Total Sentences: {num_sentences}")
        st.write(f"Average Sentence Length: {avg_sentence_length:.2f} words")
        longest_sentences = sorted(sentences, key=len, reverse=True)[:2]
        st.write("Longest Sentences:")
        st.write(longest_sentences[0])
        st.write(longest_sentences[1])
        plot_sentence_lengths(sentences)

    with tab3:
        st.header("Type-Token Ratio")
        types = set(tokens)
        ttr = len(types) / len(tokens)
        st.write(f"Type-Token Ratio (TTR): {ttr:.2f}")

    with tab4:
        st.header("Word Frequency Distribution")
        freq_dist = FreqDist(tokens)
        most_common_words = freq_dist.most_common(10)
        st.write("Top 10 Most Frequent Words:")
        for word, count in most_common_words:
            st.write(f"{word}: {count} occurrences")
        plot_word_frequency(most_common_words)

    with tab5:
        st.header("Sentiment Analysis")
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
        plot_sentiment(sentiment_scores, avg_sentiment)
