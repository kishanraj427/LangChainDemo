# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke(input="What is the capital of India?")
# print(result)


from huggingface_hub import InferenceClient
import os

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    token=api_token,
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    provider="auto" 
)
response = client.text_generation(
    prompt="What is the capital of India?",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # max_new_tokens=100,
)
print(response)




# import requests
# import os

# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# API_URL = "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# headers = {"Authorization": "Bearer "+api_token}
# payload = {"inputs": "What is the capital of India?"}

# response = requests.post(API_URL, headers=headers, json=payload)
# print(response.json())