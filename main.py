import tqdm
import os
import argparse
import numpy as np
import PyPDF2
import nltk

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CamembertModel, CamembertTokenizer

from helper import *

nltk.download('stopwords')
nltk.download('punkt')

def read_word_file(file_path):
    doc = Document(file_path)
    content = []

    for paragraph in doc.paragraphs:
        content.append(paragraph.text)

    return " ".join(content)



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

def plot(embeddings):
    
    data_list = [elmt.numpy() for elmt in embeddings.values()]
    data = np.array(data_list)
    data = data.reshape(data.shape[0], data.shape[2])

    # Initialize t-SNE with n_components=3 for three-dimensional embedding
    tsne = TSNE(n_components=2, perplexity=100, random_state=42)

    # Perform t-SNE dimensionality reduction
    embedded_data = tsne.fit_transform(data)

    plt.figure(figsize=(30, 24))  
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c='b', alpha=0.5, marker='o', edgecolors='w')  
    plt.title('t-SNE Plot for Jobs')  
    plt.xlabel('X')  
    plt.ylabel('Y')  
    plt.grid(True)  # Enable the grid
    plt.savefig('../plot.png')

parser = argparse.ArgumentParser()

parser.add_argument('--train', type=bool, default=False, help='')
parser.add_argument('--path_to_job', type=str, help='')
parser.add_argument('--K', type=int, default=10, help='The numbers of nearest cvs to compute')


args = parser.parse_args()

if __name__ == "__main__":

    model_name = "camembert-base"
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertModel.from_pretrained(model_name)

    if args.train == True:

        word_docs = '../data/word/'
        pdf_docs = '../data/pdf/'

        os.chdir(pdf_docs)
        cvs = os.listdir()
        embeddings_CV = {}
        #cvs_id = {}
        #rejected = []

        for i in range(len(cvs)):

            try:
                job_content = read_word_file(cvs[i])
                #cvs_id[i] = cvs[i]
            
                job_no_SW = remove_stopwords(job_content)
                #job_ids1 = tokenizer.encode(job1, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                job_no_SW_ids = tokenizer.encode(job_no_SW, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

                with torch.no_grad():
                    #output1 = model(job_ids1)
                    output_no_SW = model(job_no_SW_ids)

                # Extract representations from model output
                #last_hidden_states = output.last_hidden_state
                last_hidden_states = output_no_SW.last_hidden_state

                val = last_hidden_states.mean(dim=1)
                embeddings_CV[i] = val

            except Exception as e:
                #rejected.append(cvs[i])
                with open("../rejected.txt", 'a') as file:
                    file.write(f"{cvs[i]}\n")
                continue

            with open("../cv_ids.txt", 'a') as file:
                file.write(f"{i}: {cvs[i]}\n")
            
            with open("../cv_embedded.txt", 'a') as file:
                file.write(f"{i}: {val}\n")

        print("to pdf")
        #Let's go with PDF files
        #os.chdir(pdf_docs)
        cvs = os.listdir()
        #first = i + 1

        for i in range(len(cvs)):

            try:
                job_content = read_pdf_file(cvs[i])
                #cvs_id[i] = cvs[i-first]
            
                job_no_SW = remove_stopwords(job_content)
                #job_ids1 = tokenizer.encode(job1, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                job_no_SW_ids = tokenizer.encode(job_no_SW, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

                with torch.no_grad():
                    #output1 = model(job_ids1)
                    output_no_SW = model(job_no_SW_ids)

                # Extract representations from model output
                #last_hidden_states = output.last_hidden_state
                last_hidden_states = output_no_SW.last_hidden_state

                val = last_hidden_states.mean(dim=1)
                embeddings_CV[i] = val

            except Exception as e:
                #rejected.append(cvs[i])
                with open("../rejected.txt", 'a') as file:
                    file.write(f"{cvs[i]}\n")
                continue

            with open("../cv_ids.txt", 'a') as file:
                file.write(f"{55243+i}: {cvs[i]}\n")
            
            with open("../cv_embedded.txt", 'a') as file:
                file.write(f"{55243+i}: {val}\n")
        
        plot(embeddings=embeddings_CV)

    else:

        cvs = load_cvs("../data/cv_embedded.txt")

        job_content = read_pdf_file(args.path_to_job)
    
        job_no_SW = remove_stopwords(job_content)
        job_no_SW_ids = tokenizer.encode(job_no_SW, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        with torch.no_grad():
            output_no_SW = model(job_no_SW_ids)

        last_hidden_states = output_no_SW.last_hidden_state

        job = last_hidden_states.mean(dim=1)

        findings = closest(cvs, job, args.K)

        print(f"The {args.K} most likely profiles(cvs) to this job are:\n {list(findings.keys())}")
