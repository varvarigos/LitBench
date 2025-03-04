"""
Generates the related work section for a given paper.

The input
    - The input prompt is a string that contains the information of the paper for which the related work section needs to be generated.
    - The input prompt should be in the following format:
        Title of Paper: <title of the paper>

        Abstract of Paper: <abstract of the paper>
The output
    - The output is a string that contains the related work section for the given paper.
"""

import torch
import json
import networkx as nx
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import (AutoModel, AutoTokenizer, AutoModelForCausalLM)
from tqdm import tqdm
import re
import pandas as pd
import os
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import read_yaml_file


class LitFM():
    def __init__(self, graph_path):
        adapter_path = "models/llama_1b_qlora_uncensored_1_adapter_test_graph"
        config_path = 'conf/config.yaml'
        self.pretrained_model = 'BAAI/bge-large-en-v1.5'
        self.graph_name = graph_path.split('.')[0].split('/')[-1] if '/' in graph_path else graph_path.split('.')[0]
        self.batch_size = 32
        self.neigh_num = 4

        config = read_yaml_file(config_path)
        retrieval_graph_path = graph_path

        # define generation model
        model_path = config["base_model"]
        self.generation_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_tokenizer.model_max_length = 2048
        self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
        self.generation_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", )
        self.generation_model = PeftModel.from_pretrained(self.generation_model, adapter_path, adapter_name="instruction", torch_dtype=torch.float16)

        # use llama-3-8B instruct model
        self.instruction_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.instruction_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # load graph data for retrieval
        def translate_graph(graph, raw_test_nodes):
            all_nodes = list(graph.nodes())
            raw_id_2_id_dict = {}
            id_2_raw_id_dict = {}

            num = 0
            for node in all_nodes:
                raw_id_2_id_dict[node] = num
                id_2_raw_id_dict[num] = node
                num += 1
            
            new_graph = nx.Graph()
            test_edges = []
            for edge in list(graph.edges()):
                h_id, t_id = raw_id_2_id_dict[edge[0]], raw_id_2_id_dict[edge[1]]
                if edge[0] in raw_test_nodes or edge[1] in raw_test_nodes:
                    test_edges.append([h_id, t_id])
                else:
                    new_graph.add_edge(h_id, t_id)
            
            return new_graph, raw_id_2_id_dict, id_2_raw_id_dict, test_edges
        
        whole_graph_data_raw = nx.read_gexf(retrieval_graph_path, node_type=None, relabel=False, version='1.2draft')
        self.whole_graph_data, self.whole_graph_raw_id_2_id_dict, self.whole_graph_id_2_raw_id_dict, _ = translate_graph(whole_graph_data_raw, list(whole_graph_data_raw.nodes())[:1000])

        self.whole_graph_id_2_title_abs = dict()
        for paper_id in whole_graph_data_raw.nodes():
            title = whole_graph_data_raw.nodes()[paper_id]['title']
            abstract = whole_graph_data_raw.nodes()[paper_id]['abstract']
            self.whole_graph_id_2_title_abs[self.whole_graph_raw_id_2_id_dict[paper_id]] = [title, abstract]

        arxiv_topics = load_dataset("json", data_files="datasets/arxiv_topics.jsonl")
        self.whole_graph_id_2_topics = dict()
        for entry in arxiv_topics['train']:
            if entry["paper_id"] in self.whole_graph_raw_id_2_id_dict:
                self.whole_graph_id_2_topics[entry["paper_id"]] = [entry["Level 1"], entry["Level 2"], entry["Level 3"]]
        
        
        # define prompt template
        template_file_path = 'conf/alpaca.json'
        with open(template_file_path) as fp:
            self.template = json.load(fp)
        self.human_instruction = ['### Input:', '### Response:']


    def _generate_retrieval_prompt(self, data_point: dict):
        instruction =  "Please select the paper that is more likely to be cited by paper A from the given candidate papers. Your answer MUST be **only the exact title** of the selected paper. The paper title you select MUST be one of the 10 candidate papers. Do NOT generate anything else. Do NOT include explanations, formatting, or extra text. The output should be **ONLY the selected paper title** with no surrounding text and no further explanation/information."

        prompt_input = ""
        prompt_input = prompt_input + data_point['usr_prompt'] + "\n"
        prompt_input = prompt_input + "candidate papers: " + "\n"
        for i in range(len(data_point['nei_titles'])):
            prompt_input = prompt_input + str(i) + '. ' + data_point['nei_titles'][i] + "\n"
        
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res

    def _generate_sentence_prompt(self, data_point):
        instruction = "Please generate the citation sentence of how Paper A cites paper B in its related work section."

        prompt_input = ""
        prompt_input = prompt_input + data_point['usr_prompt'] + "\n"
        prompt_input = prompt_input + "Title of Paper B: " + (data_point['t_title'] if data_point['t_title'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Abstract of Paper B: " + (data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown') + "\n"

        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)

        return res

    def _generate_topic_prompt(self, data_point):
        instruction = "I need to write the related work section for this paper. Could you suggest three most relevant topics to discuss in the related work section? Your answer should be strictly one topic after the other line by line with nothing else being generated and no further explanation/information.\n"

        prompt_input = ""
        prompt_input = prompt_input + "Here are the information of the paper: \n"
        prompt_input = prompt_input + data_point['usr_prompt'] + '\n'
        prompt_input = prompt_input + "Directlty give me the topics you select.\n"
            
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res

    def _generate_paragraph_prompt(self, data_point):
        instruction = "Please write a paragraph that review the research relationships between this paper and other cited papers."
        prompt_input = ""
        prompt_input = prompt_input + data_point['usr_prompt'] + "\n"
        prompt_input = prompt_input + "Topic of this paragraph: " + data_point['topic'] + "\n"
        prompt_input = prompt_input + "papers that should be cited in paragraph: \n"

        i = data_point['paper_citation_indicator']
        for paper_idx in range(len(data_point['nei_title'])):
            prompt_input = prompt_input + "[" + str(i) + "]. " + data_point['nei_title'][paper_idx]  + '.' + " Citation sentence of this paper in the paragraph: " + data_point['nei_sentence'][paper_idx] + '\n'
            i += 1
        
        prompt_input = prompt_input + "All the above cited papers should be included and each cited paper should be indicated with its index number. Note that you should not include the title of any paper\n"
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res

    def _generate_summary_prompt(self, data_point):
        instruction = "Please combine the following paragraphs in a cohenrent way that also keeps the citations and make the flow between paragraphs more smoothly"
        instruction += "Add a sentence at the beginning of each paragraph to clarify its connection to the previous ones."
        
        prompt_input = ""
        prompt_input = prompt_input + data_point['usr_prompt'] + "\n"
        prompt_input = prompt_input + "Paragraphs that should be combined: " + "\n"
        
        i = 1
        for para in data_point['paragraphs']:
            prompt_input = prompt_input + " Paragraph " + str(i) + ": " + para + '\n'
            i += 1
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res


    @staticmethod
    def generate_text(prompt, tokenizer, model, temperature, top_p, repetition_penalty, max_new_tokens):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def get_llm_response(self, prompt, model_type):
        self.generation_model.set_adapter('instruction')
        if model_type == 'zeroshot':
            raw_output = self.generate_text(
                prompt,
                self.generation_tokenizer,
                self.generation_model,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=8096,
            )
        
        if model_type == 'zeroshot_short':
            self.generation_model.set_adapter('instruction')
            raw_output = self.generate_text(
                prompt,
                self.instruction_tokenizer,
                self.instruction_model,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=256,
            )

        if model_type == 'instruction':
            self.generation_model.set_adapter('instruction')
            raw_output = self.generate_text(
                prompt,
                self.generation_tokenizer,
                self.generation_model,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=256,
            )

        return raw_output


    def single_paper_sentence_test(self, usr_prompt, t_title, t_abs):
        datapoint = {'usr_prompt':usr_prompt, 't_title':t_title, 't_abs':t_abs}
        prompt = self._generate_sentence_prompt(datapoint)
        ans = self.get_llm_response(prompt, 'instruction')
        res = ans.strip().split(self.human_instruction[1])[-1]
        return res

    def single_paper_retrieval_test(self, usr_prompt, candidates):
        datapoint = {'usr_prompt':usr_prompt, 'nei_titles':list(candidates), 't_title': ''}
        prompt = self._generate_retrieval_prompt(datapoint)
        ans = self.get_llm_response(prompt, 'instruction')        
        res = ans.strip().split(self.human_instruction[1])[-1]
        return res

    def single_paper_topic_test(self, usr_prompt):
        datapoint = {'usr_prompt': usr_prompt}
        prompt = self._generate_topic_prompt(datapoint)
        ans = self.get_llm_response(prompt, 'zeroshot_short')
        res = ans.strip().split(self.human_instruction[1])[-1]
        return res

    def retrieval_for_one_query(self, id_2_title_abs, prompt):
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(device='cuda', dtype=torch.float16)
        model.eval()

        paper_list = list(self.whole_graph_id_2_topics.keys())
        if os.path.exists(f'datasets/{self.graph_name}_embeddings.parquet'):
            all_query_embs = torch.tensor(np.array(pd.read_parquet(f'datasets/{self.graph_name}_embeddings.parquet')))
        else:
            all_query_embs = torch.zeros(len(paper_list), 1024)
            for topic_level in ['Level 1', 'Level 2', 'Level 3']:
                i = 0
                batch_size = 2048
                candidate_emb_list = []
                pbar = tqdm(total=len(paper_list))
                while i < len(paper_list):
                    paper_batch = paper_list[i:i+batch_size]
                    paper_text_batch = []
                    for paper_id in paper_batch:
                        topics = self.whole_graph_id_2_topics[paper_id][int(topic_level[6])-1]
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

                all_query_embs += torch.cat(candidate_emb_list, 0)
    
            pd.DataFrame(all_query_embs.numpy()).to_parquet(f'datasets/{self.graph_name}_embeddings.parquet')

        # get the embeddings of the prompt
        pretrained_model_name = self.pretrained_model
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
        LLM_model.eval()

        encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
        with torch.no_grad():
            output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
            sentence_embedding = output[:, 0, :]


        tmp_scores = cosine_similarity(sentence_embedding.to("cpu"), all_query_embs.to("cpu"))[0]
        _, idxs = torch.sort(torch.tensor(tmp_scores), descending = True)
        top_10 = [int(k) for k in idxs[:10]]

        return [id_2_title_abs[i][0] for i in top_10] 


    def single_paper_related_work_generation(self, usr_prompt):    
        citation_papers = []
        nei_sentence = []

        # Get topics
        retrieval_query = self.single_paper_topic_test(usr_prompt)
        
        # Split topics
        topic_num = 3
        try:
            split_topics = retrieval_query.strip().split('\n')
            if split_topics[0] == '':
                split_topics = split_topics[1:]
            split_topics = split_topics[:topic_num]
        except:
            split_topics = retrieval_query.strip().split(':')
            split_topics = split_topics.strip().split(';')
        split_topics = split_topics[:topic_num]
        if len(split_topics) > topic_num:
            return ["too many topics", split_topics]


        # Get top-5 papers for each topic
        for retrieval_query in split_topics:
            # retrieve papers
            candidate_citation_papers = self.retrieval_for_one_query(self.whole_graph_id_2_title_abs, retrieval_query)
            
            topic_specific_citation_papers = []
            # select top-5 papers
            for i in range(5):
                # picking most likely to be cited paper
                selected_paper = self.single_paper_retrieval_test(usr_prompt, candidate_citation_papers).replace(' \n','').replace('\n','')

                words = selected_paper.strip().split(' ')
                index = -1
                for w in words:
                    try:
                        index = int(w)
                    except:
                        pass
                if index != -1 and index < len(candidate_citation_papers):
                    paper_title = candidate_citation_papers[index]
                    candidate_citation_papers = list(set(candidate_citation_papers) - set([paper_title]))
                    topic_specific_citation_papers.append(paper_title)
                else:
                    for paper_title in list(candidate_citation_papers):
                        if paper_title.lower() in selected_paper.lower() or selected_paper.lower() in paper_title.lower():
                            candidate_citation_papers = list(set(candidate_citation_papers) - set([paper_title]))
                            topic_specific_citation_papers.append(paper_title)
                            break
            citation_papers.append(topic_specific_citation_papers)

        # Generate citation sentences
        for topic_idx in range(len(citation_papers)):
            topic_specific_nei_sentence = []
            for paper_idx in range(len(citation_papers[topic_idx])):
                sentence = self.single_paper_sentence_test(usr_prompt, citation_papers[topic_idx][paper_idx], "")
                sentence = re.sub(r'\\(\S*)+\}', "", sentence)
                sentence = re.sub(r'\[(\S*)+\]', "", sentence)
                topic_specific_nei_sentence.append(sentence)
            nei_sentence.append(topic_specific_nei_sentence)

        # Generate paragraphs
        paragraphs = []
        paper_citation_indicator = 1
        for topic_idx in range(len(citation_papers)):
            datapoint = {'usr_prompt': usr_prompt, 
                        'nei_title': citation_papers[topic_idx], 
                        'nei_sentence': nei_sentence[topic_idx], 
                        'topic': split_topics[topic_idx], 
                        'paper_citation_indicator': paper_citation_indicator}
            
            prompt = self._generate_paragraph_prompt(datapoint)
            ans = self.get_llm_response(prompt, 'zeroshot')
            res = ans.strip().split(self.human_instruction[1])[-1]
            paragraphs.append(res)

            paper_citation_indicator = paper_citation_indicator + len(nei_sentence[topic_idx])

        # Generate summary
        datapoint = {'usr_prompt': usr_prompt, 'paragraphs': paragraphs}
        prompt = self._generate_summary_prompt(datapoint)
        ans = self.get_llm_response(prompt, 'zeroshot')
        summary = ans.strip().split(self.human_instruction[1])[-1]

        return summary


def gen_related_work(message, graph_path):
    LitFM_example = LitFM(graph_path)
    return LitFM_example.single_paper_related_work_generation(message)
