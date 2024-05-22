import streamlit as st
import os
import sys
import plotly.express as px
import nltk
from nltk.corpus import stopwords
import string



CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURR_DIR)

sys.path.append("".join(PARENT_DIR))

import models.openai_detector 
import models.openai_finetune_detector 
import models.radar_detector
import models.gptzero_detector


# # nltk.download('punkt')
# from nltk.util import ngrams
# from nltk.lm.preprocessing import pad_sequence, padded_everygram_pipeline
# from nltk.lm import MLE, Vocabulary
import matplotlib.pyplot as plt
# from collections import Counter
# import numpy as np

# import transformers 
# import torch
# import torch.nn.functional as F


# # add path
# from nltk.tokenize import sent_tokenize

# 
# from utils.file_helper import *
# from utils.text_helper import *
# from utils.plot_helper import *
# from models.gptzero_model import GPTZeroRunner 
# from models.detectgpt_model import DetectGPTRunner


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

def plot_bar(**kwargs):
    if 'x' not in kwargs:
        return None 
    if 'y' not in kwargs:
        return None 
    
    fig = px.bar(**kwargs)
    st.plotly_chart(fig, use_container_width=True)



# def plot_repeated_words(text):
#     tokens = preprocess_text(text)
#     word_freq = nltk.FreqDist(tokens)
#     repeated_words = [word for word, count in word_freq.items() if count > 1][:10]

#     words, counts = zip(*[(word, word_freq[word]) for word in repeated_words])

#     plt.figure(figsize=(10, 6))
#     plt.bar(words, counts)
#     plt.xlabel('Words')
#     plt.ylabel('Frequency')
#     plt.title('Repeated Words')
#     plt.xticks(rotation=45)
#     st.pyplot(plt)

# def plot_top_repeated_words(text):
#     # Tokenize the text and remove stopwords and special characters
#     tokens = text.split()
#     stop_words = set(stopwords.words('english'))
#     tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.lower() not in string.punctuation]

#     # Count the occurrence of each word
#     word_counts = Counter(tokens)

#     # Get the top 10 most repeated words
#     top_words = word_counts.most_common(10)

#     # Extract the words and their counts for plotting
#     words = [word for word, count in top_words]
#     counts = [count for word, count in top_words]

#     # Plot the bar chart using Plotly
#     fig = px.bar(x=words, y=counts, labels={'x': 'Words', 'y': 'Counts'}, title='Top 10 Most Repeated Words')
#     st.plotly_chart(fig, use_container_width=True)


def main():
    text_area = st.text_area("Enter the text you want to analyze", height=200)
    # text = st.text_area("Enter the text you want to analyze", height=200)


    if text_area is not None:
        if st.button("Check"):
            input = text_area.strip()
            
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                ######################## run openai detector                
                st.info("OpenAI RoBERTa Base")
                result_1 = models.openai_detector.detect(input)
                
                # st.write("AI-generated likelihood: ", round(result_1['Fake'], ndigits=3))
                # st.write("Human-generated likelihood: ", round(result_1['Real'], ndigits=3))
                
                # Create a slider bar
                st.slider(":robot_face:", min_value=0.0, max_value=1.0, value=float(result_1['Fake']), step=0.01, disabled=True)
                st.slider(":grinning:", min_value=0.0, max_value=1.0, value=float(result_1['Real']), step=0.01, disabled=True)

                ######################## run fine-tuned
                st.info("OpenAI RoBERTa Fine-tuned")
                result_2 = models.openai_finetune_detector.detect(input)
                st.slider(":robot_face:", min_value=0.0, max_value=1.0, value=float(result_2['Fake']), step=0.01, disabled=True)
                st.slider(":grinning:", min_value=0.0, max_value=1.0, value=float(result_2['Real']), step=0.01, disabled=True)

                ######################## run radar model
                st.info("Radar Vicuna")
                result_3 = models.radar_detector.detect(input)
                st.slider(":robot_face:", min_value=0.0, max_value=1.0, value=float(result_3['Fake']), step=0.01, disabled=True)
                st.slider(":grinning:", min_value=0.0, max_value=1.0, value=float(result_3['Real']), step=0.01, disabled=True)



            with col2:
                ######################## basic statistics
                st.info("Basic Analysis")

                tokens = preprocess_text(input)
                word_freq = nltk.FreqDist(tokens)
                most_common_words = word_freq.most_common(10)

                words, counts = zip(*most_common_words)
                args = {
                    "x": words,
                    "y": counts, 
                    "labels": {"x": "Words", "y": "Frequency"},
                    "title": "Most Common Words",
                    # "color": "continent",
                    "color_discrete_sequence": px.colors.qualitative.Pastel,
                }
                plot_bar(**args)


                
                ######################## run gptzero
                st.info("GPTZero")
                result_4 = models.gptzero_detector.detect(input)

                # Plot the bar chart using Plotly for perplexity per sentence
                perplexity_per_line_ = result_4['perplexity_per_line']
                args = {
                    "x": [i for i in range(len(perplexity_per_line_))],
                    "y": perplexity_per_line_, 
                    "labels": {"x": "Sentence", "y": "Perplexity"},
                    "title": "Perplexity per sentence",
                    "color_discrete_sequence": px.colors.qualitative.Set3,

                }
                plot_bar(**args)
                st.info(f"Perplexity: {result_4['perplexity']}, Burstiness: {result_4['burstiness']}")

                st.slider(f"Threshold: {result_4['perplexity_per_line_avg']}", 0.0, 100.0, tuple(float(x) for x in result_4['threshold']), disabled=True)
                st.write(result_4['msg'])

                # "lines": lines,

                ######################## run detectGPT

            #     # detect_gpt_runner = DetectGPTRunner('mps', 'gpt2-medium')

            #     # detect_dict = detect_gpt_runner(text_area)

            #     # st.write("Probability:", detect_dict['prob'])

            #     # st.success(detect_dict['out'])
            #     # detect_dict

            #     # tokens = nltk.corpus.brown.words()  # You can use any corpus of your choice
            #     # train_data, padded_vocab = padded_everygram_pipeline(1, tokens)
            #     # model = MLE(1)
            #     # model.fit(train_data, padded_vocab)
            #     # perplexity = calculate_perplexity(text_area, model)
            #     # burstiness_score = calculate_burstiness(text_area)

            #     # st.write("Perplexity:", perplexity)
            #     # st.write("Burstiness Score:", burstiness_score)

            #     # if perplexity > 30000 and burstiness_score < 0.2:
            #     #     st.error("Text Analysis Result: AI generated content")
            #     # else:
            #     #     st.success("Text Analysis Result: Likely not generated by AI")
                
            #     # st.warning("Disclaimer: AI plagiarism detector apps can assist in identifying potential instances of plagiarism; however, it is important to note that their results may not be entirely flawless or completely reliable. These tools employ advanced algorithms, but they can still produce false positives or false negatives. Therefore, it is recommended to use AI plagiarism detectors as a supplementary tool alongside human judgment and manual verification for accurate and comprehensive plagiarism detection.")
                
                
            # with col3:
                # st.write("Probability of AI-generated text is ", output_probs[0])
                # # plot_top_repeated_words(text_area)



if __name__ == "__main__":

    st.set_page_config(layout="wide")
    st.title("AI Text Analysis")

    main()