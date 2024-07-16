from flask import Flask, request
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModel, AutoConfig
app = Flask(__name__)

# # load model and tokenizer
# global EMBEDDING_MODEL
# global TOKENIZER
# TOKENIZER = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
# EMBEDDING_MODEL = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')


# Define the path to your desired download location (ensure it has enough space)
# cache_dir = "/cache"  # Replace with your actual path

# # Load the model with the specified cache directory
# global MODEL
# config = AutoConfig.from_pretrained("Salesforce/SFR-Embedding-Mistral", cache_dir=cache_dir)
# MODEL = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral", config=config, cache_dir=cache_dir)


# EMBEDDING_MODEL = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@app.route("/", methods=["GET"])
def hello():
    return "Embedding Service powdered by Salesforce/SFR-Embedding-Mistral."


@app.route("/embed", methods=["POST"])
def embed_string():
    data = request.get_json()
    query = " ".join(data["text"].strip().split())
    print("query received", query)
    passages = [query]
    # get the embeddings
    max_length = 4096
    # input_texts = queries + passages
    # input_texts = passages
    # batch_dict = TOKENIZER(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    # outputs = EMBEDDING_MODEL(**batch_dict)
    # embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # # normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)

    # # try:
    # #     embeddings = EMBEDDING_MODEL.encode(query).tolist()
    # # except Exception as e:
    # #     print("Exception occcured while embedding: ", e)
    # #     embeddings = []
    embeddings = []
    return {"len_of_embedding" : len(embeddings), "embeddings": embeddings}

if __name__ == "__main__":
    app.run(debug=True)

