from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

text = "The capital of India is New Delhi."
vector = embeddings.embed_query(text)

print(str(vector))
print(len(vector), vector[:5])  
