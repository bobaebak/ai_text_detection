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
import models.detectgpt_detector


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

                ######################## run openai detector                
                st.info("OpenAI RoBERTa Large")
                result_1 = models.openai_detector.detect(input)
                
                # st.write("AI-generated likelihood: ", round(result_1['Fake'], ndigits=3))
                # st.write("Human-generated likelihood: ", round(result_1['Real'], ndigits=3))
                
                # Create a slider bar
                st.slider(":robot_face:", min_value=0.0, max_value=1.0, value=float(result_1['Fake']), step=0.01, disabled=True)
                st.slider(":grinning:", min_value=0.0, max_value=1.0, value=float(result_1['Real']), step=0.01, disabled=True)

                ######################## run fine-tuned
                st.info("OpenAI RoBERTa Fine-tuned")
                result_2 = models.openai_finetune_detector.detect(input, None, None, "20240523_v1_epoch4.pth")
                st.slider(":robot_face:", min_value=0.0, max_value=1.0, value=float(result_2['Fake']), step=0.01, disabled=True)
                st.slider(":grinning:", min_value=0.0, max_value=1.0, value=float(result_2['Real']), step=0.01, disabled=True)

                ######################## run radar model
                st.info("Radar Vicuna")
                result_3 = models.radar_detector.detect(input)
                st.slider(":robot_face:", min_value=0.0, max_value=1.0, value=float(result_3['Fake']), step=0.01, disabled=True)
                st.slider(":grinning:", min_value=0.0, max_value=1.0, value=float(result_3['Real']), step=0.01, disabled=True)



            with col2:
                ######################## run gptzero
                st.info("GPTZero")
                result_4 = models.gptzero_detector.detect(input)
                st.slider(f"Threshold: {result_4['perplexity_per_line_avg']}", 0.0, 100.0, tuple(float(x) for x in result_4['threshold']), disabled=True)
                st.markdown(f"- *Threshold < {result_4['threshold'][0]}:* The Text is generated by AI")
                st.markdown(f"- *{result_4['threshold'][0]} <= Threshold < {result_4['threshold'][1]}:* The Text is most probably contain parts which are generated by AI")
                st.markdown(f"- *{result_4['threshold'][1]} >= Threshold:* The Text is written by Human")
                
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
                st.success(f"""{result_4['msg']} \n
                           Perplexity: {result_4['perplexity']}, Burstiness: {result_4['burstiness']}""")


                # ######################## run DetectGPT
                # st.info("DetectGPT")
                # result_5 = models.detectgpt_detector.detect(input)
                # st.slider(f"Threshold: {result_5['mean_score']}", 0.0, 1.0, float(result_5['threshold']), disabled=True)
                # st.markdown(f"- *Threshold >= {result_5['threshold']}:* The Text is generated by AI")
                # st.markdown(f"- *Threshold < {result_5['threshold']}:* The Text is most likely written by Human")
                # st.success(f"""{result_5['msg']}""")

            with col3:
                ######################## run GAN version
                st.info("GAN")
                # st.write("Probability of AI-generated text is ", output_probs[0])
                # # plot_top_repeated_words(text_area)
                pass

if __name__ == "__main__":

    st.set_page_config(layout="wide")
    st.title("AI Text Analysis")

    main()