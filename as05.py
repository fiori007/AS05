import gdown
import PyPDF2
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
import re
import streamlit as st

# key
openai.api_key = os.getenv('OPENAI_API_KEY')

# url do arquivo pdf no drive
url = 'https://drive.google.com/uc?id=1DzmmRVUcLL5aaEv57Q7XsPmNNNpTpg7E'
pdf_file_path = 'document.pdf'  # Atualizado para o contexto do Streamlit

# Baixa o PDF do Drive
gdown.download(url, pdf_file_path, quiet=False)

# Função para extrair o texto do PDF com limpeza
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    # Limpeza de caracteres desnecessários
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text.strip()

# Parágrafos menores
def split_text_into_chunks(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Verificação de extração correta do texto
pdf_text = extract_text_from_pdf(pdf_file_path)
chunks = split_text_into_chunks(pdf_text, chunk_size=500)  # Dividindo em chunks

# Carregar modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gerar embeddings para o texto do PDF
pdf_embeddings = model.encode(chunks)

# Função que processa as perguntas e obtém respostas usando o GPT
def ask_gpt(question):
    messages = [{"role": "user", "content": question}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Modelo de linguagem do GPT
        messages=messages,
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

# Função para encontrar o trecho mais similar no texto com base nos embeddings
def find_similar_section(question, pdf_embeddings):
    question_embedding = model.encode([question])
    similarities = np.dot(pdf_embeddings, question_embedding.T)
    most_similar_idx = np.argmax(similarities)
    similar_text = chunks[most_similar_idx]
    return similar_text

# Função principal do chatbot
def chatbot_with_embeddings(question):
    similar_section = find_similar_section(question, pdf_embeddings)
    response = ask_gpt(f"Baseado no texto: {similar_section}, responda a seguinte pergunta: {question}")
    return response

# Interface do Streamlit
st.title("Assistente Conversacional sobre PDF")
st.write("Faça perguntas sobre o conteúdo do PDF e obtenha respostas.")

# Campo de entrada para perguntas
question = st.text_input("Digite sua pergunta:")

# Botão para enviar pergunta
if st.button("Obter resposta"):
    if question:
        response = chatbot_with_embeddings(question)
        st.write("Resposta do GPT:")
        st.write(response)
    else:
        st.write("Por favor, insira uma pergunta.")
