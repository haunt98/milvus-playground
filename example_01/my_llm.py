from transformers import AutoModel, AutoTokenizer
import torch

# https://github.com/QwenLM/Qwen2.5
# https://huggingface.co/Qwen/Qwen2.5-0.5B
model_name = "Qwen/Qwen2.5-0.5B"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()

print(f"Model {model_name} embedding dimension: {model.config.hidden_size}")


# https://milvus.io/docs/integrate_with_hugging-face.md
def generate_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding
