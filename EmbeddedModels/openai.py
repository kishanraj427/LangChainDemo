from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from torch import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

# for single embedding
embeddingResult = embedding.embed_query("Delhi is the capital of India")
print(str(embeddingResult))

# for multiple embedding
document = [
    "Delhi is the capital of India", 
    "Ranchi is the capital of Jharkhand",
    "Patna is the capital of Bihar"
]
embeddingResult = embedding.embed_documents(document)
print(str(embeddingResult))

# finding the similarity

query = "Tell me about Ranchi?"
queryEmbedding = embedding.embed_query(query)

scores = cosine_similarity([queryEmbedding], embeddingResult)
index, score =  sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(document[index])
print("Similarity score is ", score)
