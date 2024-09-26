from pymilvus import MilvusClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

client = MilvusClient(
    uri="http://localhost:19530",  # replace with your own Milvus server address
)

collection_name = "maverick"

collection_load_state = client.get_load_state(
    collection_name=collection_name,
)

print(f"Collection {collection_name} loaded state: {collection_load_state}")

# TODO: Support mps if available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
print(f"Device: {device}")

# https://github.com/QwenLM/Qwen2.5
# https://huggingface.co/Qwen/Qwen2.5-0.5B
model_name = "Qwen/Qwen2.5-0.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

print(f"Model {model_name} embedding dimension: {model.config.hidden_size}")


# https://milvus.io/docs/integrate_with_hugging-face.md
def generate_embedding(text):
    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    token_embeddings = model_output[0]
    attention_mask = encoded_input["attention_mask"]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sentence_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Normalize embeddings
    embedding = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return embedding


filename_data = "data_tiny.txt"
with open(filename_data, "r") as f:
    data = f.readlines()

vector_data = [
    {"id": i, "vector": generate_embedding(data[i]), "text": data[i]}
    for i in range(len(data))
]

# DEBUG
# print(vector_data)
