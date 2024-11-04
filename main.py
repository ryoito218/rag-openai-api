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

question = "2023年の第1事業部の売上はどのくらい？"

documents = [
    "2023年上期売上200億円，下期売上300億円",
    "2023年第1事業部売上300億円，第2事業部売上150億円，第3事業部売上50億円",
    "2024年は全社で1000億円の売り上げを目指す"
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

print(documents[most_similar_index])