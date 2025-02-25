# arXiv Topic-Level Embeddings Dataset

## Dataset Summary

The arXiv Topic-Level Embeddings Dataset provides embedding representations for topics associated with arXiv papers. Specifially, the dataset contains embeddings of the [arXiv Topics Dataset](https://www.example.com) repository and is used at the retriever module of LitBench to identify relevant papers based on user queries by calculating the similarity bwteen these paper embeddings and the embedding representation of the user query. These embeddings were generated using the BAAI/bge-large-en-v1.5 model. The dataset consists of 2,422,486 paper IDs, each mapped to a single embedding.

These embeddings can be used for document retrieval, semantic search, topic modeling, and clustering.

## Dataset Structure

### Data Fields

Each row in the dataset contains:
'''python
{
  "paper_id": "2401.12345",
  "embedding": [0.12, -0.43, 0.87, ...]
}
'''

- paper_id: Unique identifier for the paper (following arXiv ID format).

- embedding: A numerical vector representing the topic-related semantics.

The dataset is stored in parquet format for efficient querying and processing.

## Usage

To load the dataset using pandas:

```python
import pandas as pd

# Load Level 1 embeddings
papers_df = pd.read_parquet("arxiv_papers_embeds.parquet")

# Retrieve the first paper's embedding
sample_paper = papers_df.iloc[0]
print(f"paper_id: {sample_paper['paper_id']}")
print(f"Embedding: {sample_paper['embedding']}")
'''

You can use these embeddings to calculate similarity between papers and research topics. For example, the following code snippet will calculate the cosine similarity between two papers:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import pandas as pd

paper_embeds = torch.tensor(np.array(pd.read_parquet("arxiv_papers_embeds.parquet")['embedding'].tolist()))

user_topic = "Machine Learning"

model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device='cuda', dtype=torch.float16)
inputs = tokenizer([query], return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs.to('cuda'))
    query_embeddings = outputs.last_hidden_state[:, 0, :].cpu()


similarity = cosine_similarity(query_embeddings, paper_embeds)[0]
print(f"Cosine Similarity: {similarity}")
'''

For processing with Hugging Face datasets:

```python
from datasets import load_dataset

dataset = load_dataset("your-huggingface-username/arXiv-Topic-Embeddings")
print(dataset["train"][0])
'''

This dataset is particularly useful for semantic search, information retrieval, and large-scale text analysis.

## Citation

If you use the LitBench Topic-Level Embeddings Dataset in your research, please cite our work:

```bibtex
@misc{litbench2025embeddings,
  title={LitBench: A Large Language Model Benchmarking Framework For Literature Tasks},
  author={Your Name and Co-authors},
  year={2025}
}
'''

For further details about the LitBench framework, refer to our main repository: [LitBench GitHub](https://www.example.com).
