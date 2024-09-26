from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",  # replace with your own Milvus server address
)

collection_name = "maverick"
dimension = 768

if client.has_collection(collection_name=collection_name):
    print(f"Collection {collection_name} already exists.")
else:
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )


collection_load_state = client.get_load_state(
    collection_name=collection_name,
)

print(f"Collection {collection_name} loaded state: {collection_load_state}")
