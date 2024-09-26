from pymilvus import MilvusClient
from pymilvus import model

client = MilvusClient("milvus_demo.db")

embedding_fn = model.DefaultEmbeddingFunction()

query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)
