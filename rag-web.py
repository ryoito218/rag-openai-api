import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text_nodes = soup.find_all("div")
    joined_text = "".join(t.text.replace("\t", "").replace("\n", "") for t in text_nodes)
    
    return joined_text

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start + chunk_size <= len(text):
        chunks.append(text[start:start+chunk_size])
        start += (chunk_size - overlap)

    if start < len(text):
        chunks.append(text[-chunk_size:])

    return chunks

def vectorize_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def find_most_similar(question_vector, vectors, documents):
    similarities = []

    for index, vector in enumerate(vectors):
        similarity = cosine_similarity([question_vector], [vector])[0][0]
        similarities.append([similarity, index])
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_documents = [documents[index] for similarity, index in similarities[:2]]
    
    return top_documents


    # max_similarity = 0
    # most_similar_index = 0

    # for index, vector in enumerate(vectors):
    #     similarity = cosine_similarity([question_vector], [vector])[0][0]
    #     if similarity > max_similarity:
    #         max_similarity = similarity
    #         most_similar_index = index
    
    # return documents[most_similar_index]

def ask_question(question, context):
    prompt = f"""以下の質問に以下の情報をベースにして答えてください
    # ユーザの質問
    {question}

    # 情報
    {context}
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200
    )

    return response.choices[0].text

url = "https://XXX.com"
chunk_size = 400
overlap = 50

article_text = scrape_article(url)
text_chunks = chunk_text(article_text, chunk_size, overlap)

vectors = [ vectorize_text(doc) for doc in text_chunks]

question = "質問を入力"
question_vector = vectorize_text(question)

similar_document = find_most_similar(question_vector, vectors, text_chunks)

answer = ask_question(question, similar_document)

print(answer)