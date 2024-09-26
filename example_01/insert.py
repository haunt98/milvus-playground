from pymilvus import MilvusClient
from my_llm import generate_embedding

client = MilvusClient(
    uri="http://localhost:19530",  # replace with your own Milvus server address
)

collection_name = "maverick"

collection_load_state = client.get_load_state(
    collection_name=collection_name,
)

print(f"Collection {collection_name} loaded state: {collection_load_state}")

filename_data = "data_tiny.txt"
with open(filename_data, "r") as f:
    data = f.readlines()

print("Data has", len(data), "entities")

vector_data = [
    {"id": i, "vector": generate_embedding(data[i]), "text": data[i]}
    for i in range(len(data))
]

# DEBUG
# print(vector_data)

insert_rsp = client.insert(collection_name=collection_name, data=vector_data)
print(insert_rsp)
