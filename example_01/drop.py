from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",  # replace with your own Milvus server address
)

collection_name = "maverick"

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)
