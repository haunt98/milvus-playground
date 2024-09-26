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


query_vectors = [generate_embedding("Chính phủ Việt Nam")]

search_rsp = client.search(
    collection_name=collection_name,  # target collection
    data=query_vectors,  # query vectors
    limit=3,  # number of returned entities
    output_fields=["text"],  # specifies fields to be returned
)
print(search_rsp)
