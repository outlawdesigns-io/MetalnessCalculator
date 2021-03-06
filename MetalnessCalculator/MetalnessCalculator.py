import os
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import scipy
import string
from collections import Counter
from nltk import FreqDist, tokenize

class MetalnessCalculator:
    SWEAR_DATA = os.path.dirname(os.path.realpath(__file__)) + '/../data/swear_words_eng.txt'
    STOP_DATA = os.path.dirname(os.path.realpath(__file__)) + '/../data/stopwords_eng.txt'
    def __init__(self,metalDf,controlDf):
        self.setData(metalDf,controlDf)
        self.train()
    def setData(self,metalDf,controlDf):
        self.metalDf = metalDf
        self.controlDf = controlDf
    def train(self):
        self.SWEAR_WORDS = [str(line.rstrip('\n')) for line in open(self.SWEAR_DATA, "r")]
        self.STOPWORDS = list(set([str(line.rstrip('\n')) for line in open(self.STOP_DATA, "r")]))
        self.PUNCTUATION =  list(string.punctuation) + ['..', '...', '’', "''", '``', '`']
        self.metal_word_freq_dist = self.get_word_frequence_distribution(self.metalDf, 'lyrics')
        self.no_metal_word_freq_dist = self.get_word_frequence_distribution(self.controlDf,'lyrics')
        self.words_metalness_df = self.calculate_words_metalness(self.metal_word_freq_dist, self.no_metal_word_freq_dist)\
        .sort_values(['metalness'], ascending=False).reset_index().drop(columns=['index'])
    def get_word_frequence_distribution(self,df, text_column):
        words_corpus = " ".join(df[text_column].astype(str).values)
        words_corpus = words_corpus.lower().replace('\\n', ' ')
        word_freq_dist = FreqDist(nltk.word_tokenize(words_corpus))
        # remove punctuation and stopwords
        for stopword in self.STOPWORDS:
            if stopword in word_freq_dist:
                del word_freq_dist[stopword]
        for punctuation in self.PUNCTUATION:
            if punctuation in word_freq_dist:
                del word_freq_dist[punctuation]
        return word_freq_dist
    def calculate_words_metalness(self,metal_wfd, no_metal_wfd):
        no_metal_wfd = {k:v for k,v in no_metal_wfd.items() if v >= 5}
        num_no_metal_words = sum(no_metal_wfd.values())
        metal_wfd = {k:v for k,v in metal_wfd.items() if v >= 5}
        num_metal_words = sum(metal_wfd.values())
        metalness = {}
        for w in metal_wfd.keys() & no_metal_wfd.keys():
            if len(w) > 2:
                metal_coefficient = math.log((metal_wfd[w] / num_metal_words) / (no_metal_wfd[w] / num_no_metal_words))
                metalness[w] = 1 / (1 + math.exp(-metal_coefficient / 2))

        metalness_df = pd.DataFrame({
            'words': list(metalness.keys()),
            'metalness': list(metalness.values())
        })
        return metalness_df
    def calculate_metalness_score(self,lyrics):
        lyrics = lyrics.lower().replace('\\n', ' ').strip()
        words = nltk.word_tokenize(lyrics)
        words = [word for word in words if word not in self.PUNCTUATION]
        words = [word for word in words if word not in self.STOPWORDS]
        if len(words) == 0:
            return 0
        song_score = 0
        for word in words:
            if len(self.words_metalness_df.loc[self.words_metalness_df['words'] == word, 'metalness'].values) == 0:
                song_score += 0
            else:
                song_score += float(self.words_metalness_df.loc[self.words_metalness_df['words'] == word, 'metalness'].values[0])
        return song_score / len(words)
