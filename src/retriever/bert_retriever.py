from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
import torch
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from datasets import load_dataset


def generate_topic_level_embeddings(model, tokenizer, paper_list, tmp_id_2_abs):
    id2topics = {
        entry["paper_id"]: [entry["Level 1"], entry["Level 2"], entry["Level 3"]]
        for entry in tmp_id_2_abs
    }

    for topic_level in ['Level 1', 'Level 2', 'Level 3']:
        i = 0
        batch_size = 2048
        candidate_emb_list = []
        pbar = tqdm(total=len(paper_list))
        while i < len(paper_list):
            yield i / len(paper_list) / 3 if topic_level == 'Level 1' else 0.33 + i / len(paper_list) / 3 if topic_level == 'Level 2' else 0.66 + i / len(paper_list) / 3
            paper_batch = paper_list[i:i+batch_size]
            paper_text_batch = []
            for paper_id in paper_batch:
                topics = id2topics[paper_id][int(topic_level[6])-1]
                topic_text = ''
                for t in topics:
                    topic_text += t + ','
                paper_text_batch.append(topic_text)
            inputs = tokenizer(paper_text_batch, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs.to('cuda'))
                candidate_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                candidate_embeddings = candidate_embeddings.reshape(-1, 1024)
                candidate_emb_list.append(candidate_embeddings)

                i += len(candidate_embeddings)
                pbar.update(len(candidate_embeddings))

        all_candidate_embs = torch.cat(candidate_emb_list, 0)
        
        df = pd.DataFrame({
            "paper_id": paper_list,
            "embedding": list(all_candidate_embs.numpy())
        })
        
        if not os.path.exists('datasets/topic_level_embeds'):
            os.makedirs('datasets/topic_level_embeds')

        df.to_parquet(f'datasets/topic_level_embeds/{topic_level}_emb.parquet', engine='pyarrow', compression='snappy')
        
    all_candidate_embs_L1 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 1_emb.parquet')['embedding'].tolist())).half()
    all_candidate_embs_L2 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 2_emb.parquet')['embedding'].tolist())).half()
    all_candidate_embs_L3 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 3_emb.parquet')['embedding'].tolist())).half()
    all_candidate_embs = all_candidate_embs_L1 + all_candidate_embs_L2 + all_candidate_embs_L3
    
    df = pd.DataFrame({
        "paper_id": paper_list,
        "embedding": list(all_candidate_embs.numpy())
    })
    
    df.to_parquet('datasets/topic_level_embeds/arxiv_papers_embeds.parquet', engine='pyarrow', compression='snappy')
    


def retriever(query, retrieval_nodes_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    yield 0
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device='cuda', dtype=torch.float16)
    inputs = tokenizer([query], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs.to('cuda'))
        query_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    
    # Load the dataset
    tmp_id_2_abs = load_dataset("json", data_files="datasets/arxiv_topics.jsonl")
    tmp_id_2_abs = tmp_id_2_abs['train']
    paper_list = list(tmp_id_2_abs['paper_id'])
    
    # if the file does not exist
    if not os.path.exists('datasets/topic_level_embeds/arxiv_papers_embeds.parquet'):
        yield from generate_topic_level_embeddings(model, tokenizer, paper_list, tmp_id_2_abs)
        
    all_candidate_embs = load_dataset("parquet", data_files="datasets/topic_level_embeds/arxiv_papers_embeds.parquet")['train']


    # Calculate the cosine similarity between the query and all candidate embeddings    
    similarity_scores = cosine_similarity(query_embeddings, all_candidate_embs)[0]
    

    # Sort the papers by similarity scores and select the top K papers
    id_score_list = []
    for i in range(len(paper_list)):
        id_score_list.append([paper_list[i], similarity_scores[i]])
    
    sorted_scores = sorted(id_score_list, key=lambda i: i[-1], reverse = True)
    top_K_paper = [sample[0] for sample in sorted_scores[:30000]]
    print(top_K_paper)

    papers_results = {
        paper: True
        for paper in top_K_paper
    }

    with open(retrieval_nodes_path, 'w') as f:
        json.dump(papers_results, f)
    
    yield 1.0
