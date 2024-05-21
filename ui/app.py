import streamlit as st
import nltk
# nltk.download('punkt')
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline
from nltk.lm import MLE, Vocabulary
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
import plotly.express as px
from collections import Counter
import numpy as np


# add path
from nltk.tokenize import sent_tokenize

import sys
sys.path.append("/Users/bobaebak/git/ai_text_detection")
from utils.file_helper import *
from utils.text_helper import *
from utils.plot_helper import *
from models.gptzero_model import GPTZeroRunner 

st.set_page_config(layout="wide")


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

def plot_most_common_words(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)
    most_common_words = word_freq.most_common(10)

    words, counts = zip(*most_common_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def plot_repeated_words(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)
    repeated_words = [word for word, count in word_freq.items() if count > 1][:10]

    words, counts = zip(*[(word, word_freq[word]) for word in repeated_words])

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Repeated Words')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def calculate_perplexity(text, model):
    tokens = preprocess_text(text)
    padded_tokens = ['<s>'] + tokens + ['</s>']
    ngrams_sequence = list(ngrams(padded_tokens, model.order))
    perplexity = model.perplexity(ngrams_sequence)
    return perplexity

def plot_top_repeated_words(text):
    # Tokenize the text and remove stopwords and special characters
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

    # Count the occurrence of each word
    word_counts = Counter(tokens)

    # Get the top 10 most repeated words
    top_words = word_counts.most_common(10)

    # Extract the words and their counts for plotting
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]

    # Plot the bar chart using Plotly
    fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Top 10 Most Repeated Words')
    st.plotly_chart(fig, use_container_width=True)



def calculate_burstiness(text):
    tokens = preprocess_text(text)
    word_freq = nltk.FreqDist(tokens)

    avg_freq = sum(word_freq.values()) / len(word_freq)
    variance = sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq)

    burstiness_score = variance / (avg_freq ** 2)
    return burstiness_score

def is_generated_text(perplexity, burstiness_score):
    if perplexity < 100 and burstiness_score < 1:
        return "Likely generated by a language model"
    else:
        return "Not likely generated by a language model"


def main():
    st.title("AI Text Analysis")
    text_area = st.text_area("Enter text", "")

    if text_area is not None:
        if st.button("Check"):
            text_area = text_area.strip()
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.info("GPTZero Score")
                
                model = GPTZeroRunner('mps', 'gpt2-medium')
                ppl_dict = model(text_area)

                sentences = sent_tokenize(text_area)
                word_counts = [len(sentence.split()) for sentence in sentences]
                avg_word_count = round(np.mean(word_counts), 1)

                st.write("Perplexity:", ppl_dict['perplexity_per_line_score'])
                st.write("Burstiness:", ppl_dict['burstiness'])
                st.write("Average Sentence Length:", avg_word_count, "words")

                # Create a slider bar
                # slider_value = st.slider("Average Sentence Length", min_value=0.0, max_value=100.0, value=avg_word_count, step=0.1)

                st.success(ppl_dict['out'])
                ppl_dict

            
            with col2:
                st.info("Detection Score")
                tokens = nltk.corpus.brown.words()  # You can use any corpus of your choice
                train_data, padded_vocab = padded_everygram_pipeline(1, tokens)
                model = MLE(1)
                model.fit(train_data, padded_vocab)
                perplexity = calculate_perplexity(text_area, model)
                burstiness_score = calculate_burstiness(text_area)

                st.write("Perplexity:", perplexity)
                st.write("Burstiness Score:", burstiness_score)

                if perplexity > 30000 and burstiness_score < 0.2:
                    st.error("Text Analysis Result: AI generated content")
                else:
                    st.success("Text Analysis Result: Likely not generated by AI")
                
                st.warning("Disclaimer: AI plagiarism detector apps can assist in identifying potential instances of plagiarism; however, it is important to note that their results may not be entirely flawless or completely reliable. These tools employ advanced algorithms, but they can still produce false positives or false negatives. Therefore, it is recommended to use AI plagiarism detectors as a supplementary tool alongside human judgment and manual verification for accurate and comprehensive plagiarism detection.")
                
                
            with col3:
                st.info("Basic Details")
                plot_top_repeated_words(text_area)


    # text = st.text_area("Enter the text you want to analyze", height=200)
    # if st.button("Analyze"):
        # if text:
        #     # Load or train your language model
        #     # In this example, we'll use a simple unigram model
        #     tokens = nltk.corpus.brown.words()  # You can use any corpus of your choice
        #     train_data, padded_vocab = padded_everygram_pipeline(1, tokens)
        #     model = MLE(1)
        #     model.fit(train_data, padded_vocab)

        #     # Calculate perplexity
        #     perplexity = calculate_perplexity(text, model)
        #     st.write("Perplexity:", perplexity)

        #     # Calculate burstiness score
        #     burstiness_score = calculate_burstiness(text)
        #     st.write("Burstiness Score:", burstiness_score)

        #     # Check if text is likely generated by a language model
        #     generated_cue = is_generated_text(perplexity, burstiness_score)
        #     st.write("Text Analysis Result:", generated_cue)
            
        #     # Plot most common words
        #     plot_most_common_words(text)

        #     # Plot repeated words
        #     plot_repeated_words(text)
            
        # else:
        #     st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()