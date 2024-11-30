
import gdown
import PyPDF2
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
import re
import gradio as gr

# key
openai.api_key = os.getenv('OPENAI_API_KEY')

# url do arquivo pdf no drive
url = 'https://drive.google.com/uc?id=1DzmmRVUcLL5aaEv57Q7XsPmNNNpTpg7E'
pdf_file_path = '/content/document.pdf'

# baixa o pdf do Drive
gdown.download(url, pdf_file_path, quiet=False)

# função para extrair o texto do pdf com limpeza
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    # caracteres desenecessários
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text.strip()

# parágrafos menores
def split_text_into_chunks(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# verificaçao de foi extraido corretamente o texto
pdf_text = extract_text_from_pdf(pdf_file_path)
chunks = split_text_into_chunks(pdf_text, chunk_size=500)  # Dividindo em chunks

# modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

pdf_embeddings = model.encode(chunks)

# função que processa as perguntas e obtem respostas usando o GPT
def ask_gpt(question):
    # usando o endpoint correto para o modelo de chat
    messages = [{"role": "user", "content": question}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  #  modelo de linguagem do GPT
        messages=messages,
        max_tokens=150
    )
    return response['choices'][0]['message']['content'].strip()

def find_similar_section(question, pdf_embeddings):
    # gera embedding para a pergunta
    question_embedding = model.encode([question])

    # calcula a similaridade entre a pergunta e os embeddings do pdf
    similarities = np.dot(pdf_embeddings, question_embedding.T)
    most_similar_idx = np.argmax(similarities)

    # retorna o trecho mais similar
    similar_text = chunks[most_similar_idx]
    return similar_text

# Função que utiliza embeddings
def chatbot_with_embeddings(question):
    similar_section = find_similar_section(question, pdf_embeddings)
    response = ask_gpt(f"Baseado no texto: {similar_section}, responda a seguinte pergunta: {question}")
    return response

# interface Gradio
def gradio_interface(question):
    return chatbot_with_embeddings(question)

# criando a interface com Gradio
interface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text",
                         title="Assistente Conversacional sobre PDF",
                         description="Faça perguntas sobre o conteúdo do PDF e obtenha respostas.")

interface.launch() # executa a interface