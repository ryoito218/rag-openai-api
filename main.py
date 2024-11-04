from openai import OpenAI
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def vectorize_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

question = "質問を入力"

documents = [
    "ドキュメント1",
    "ドキュメント2",
    "ドキュメント3"
]

vectors = [vectorize_text(doc) for doc in documents]
question_vector = vectorize_text(question)

max_similarity = 0
most_similar_index = 0

for index, vector in enumerate(vectors):
    similarity = cosine_similarity([question_vector], [vector])[0][0]
    print(documents[index], ":", similarity)
    if similarity > max_similarity:
        max_similarity = similarity
        most_similar_index = index

prompt = f"""以下の質問に以下の情報をベースにして答えてください
# ユーザの質問
{question}

# 情報
{documents[most_similar_index]}
"""

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=200
)

print(response.choices[0].text)