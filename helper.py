import torch
import torch.nn as nn
import numpy as np

import torch
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

from sklearn.metrics.pairwise import cosine_similarity
from transformers import CamembertModel, CamembertTokenizer

def closest(cvs, job, K):
    #Return the ids of the k-nearest neighboors to job from the set of cvs
    
    knns = None
    cos = nn.CosineSimilarity()
    distances = {i: cos(job, elmt) for i, elmt in cvs.items()}
    sorted_distances = sorted(distances.items(), key=lambda x:x[1], reverse=True)
    knns = dict(sorted_distances[:K])

    return knns

def load_cvs(path):
    #Load the dict of cvs and return it in a dict
    
    with open(path) as file:
        text = file.read()
        file.close()
    
    cv_embed = text.split("]])\n")
    cv_embed[-1] = cv_embed[-1].replace("]])", "")
    cv_embed = [elmt.replace("\n         ", "  ").replace(",  ", ", ").replace(", ", " ").replace(": tensor([[ ", ": tensor([[") for elmt in cv_embed]
    cv_embed.remove('')
    
    embeddings = {}

    for elmt in cv_embed:
        id, value = tuple(elmt.split(": tensor([["))
        value = value.replace('  ', ' ')
        val = [float(elmt) for elmt in value.split(' ')]
        embeddings[id] = torch.tensor(val)

    return embeddings    

def read_pdf_file(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text_content = []

        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text_content.append(page.extract_text())

    return " ".join(text_content)


def remove_stopwords(text):
    # Tokenize the text into words
    stop_words = set(stopwords.words('french'))
    words = word_tokenize(text)

    # Remove stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Reconstruct the text without stop words
    filtered_text = ' '.join(filtered_words)

    return filtered_text

