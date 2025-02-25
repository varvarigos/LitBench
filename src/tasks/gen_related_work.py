import pickle
import torch
import json
import yaml
import networkx as nx
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import (AutoConfig, AutoModel, AutoTokenizer, pipeline, AutoModelForCausalLM)

# from navigator.Dataset import NodeDataset_QA, TestEdgeDataset_QA
# from navigator.Model import retriever
from retriever.bert_retriever import retriever
from torch.utils.data import DataLoader 

from tqdm import tqdm
import re
import pandas as pd


def NodeDataset_QA(graph, id_2_title_abs, neigh_num, tokenizer):
    all_nodes = list(graph.nodes())
    all_edges = list(graph.edges())
    all_edges = [[all_nodes.index(edge[0]), all_nodes.index(edge[1])] for edge in all_edges]
    all_edges += [[edge[1], edge[0]] for edge in all_edges]
    all_edges = list(set([tuple(edge) for edge in all_edges]))
    all_edges = [[edge[0], edge[1]] for edge in all_edges]

    all_edges = torch.LongTensor(all_edges)
    all_edges = all_edges.t().contiguous()
    all_edges = all_edges.cuda()

    all_nodes = torch.LongTensor(all_nodes)
    all_nodes = all_nodes.cuda()

    all_edges = all_edges.t().contiguous()
    all_edges = all_edges.cuda()

    all_nodes = all_nodes.cuda()

    return {'graph': graph, 'id_2_title_abs': id_2_title_abs, 'neigh_num': neigh_num, 'tokenizer': tokenizer, 'all_edges': all_edges, 'all_nodes': all_nodes}

def TestEdgeDataset_QA(tokenizer, test_query_embs):
    return {'tokenizer': tokenizer, 'test_query_embs': test_query_embs}

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

class LitFM():
    def __init__(self):
        adapter_path = "models/llama_1b_qlora_uncensored_1_adapter_test_graph"
        retrieval_graph_path = "datasets/quantum_graph.gexf"
        config_path = 'conf/config.yaml'
        self.pretrained_model = 'BAAI/bge-large-en-v1.5'
        self.graph_name = 'quantum'
        self.batch_size = 32
        self.neigh_num = 4

        
        config = read_yaml_file(config_path)
        model_family = config["model_family"]
        # define generation model
        if model_family=='llama':
            model_path = config["base_model"]
            access_token = config["huggingface"]["inference_token"]
            self.generation_tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
            self.generation_tokenizer.model_max_length = 2048
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token
            self.generation_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", token=access_token)
        if model_family=='mistral' or model_family=='vicuna':
            model_path = config["base_model"]
            self.generation_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.generation_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=False)
        
        self.generation_model = PeftModel.from_pretrained(self.generation_model, adapter_path, adapter_name="instruction")

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
        print([len(self.whole_graph_data.nodes()), len(self.whole_graph_data.edges())])

        # define retrieval model
        retrieval_config = AutoConfig.from_pretrained(self.pretrained_model, output_hidden_states=True)
        retrieval_config.heter_embed_size = 256
        self.hidden_size = retrieval_config.hidden_size
        self.retrieval_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.retrieval_model = retriever(config, self.hidden_size).cuda() #torch.load('/home/jz875/project/Citation_Graph/LLM_3_37/navigator/save_best_1211bert-base-uncased_CS.pt').cuda() #retriever(config, args).cuda()
        
        # define prompt template
        template_file_path = 'conf/alpaca.json'
        with open(template_file_path) as fp:
            self.template = json.load(fp)
        self.human_instruction = ['### Input:', '### Response:']


    def _generate_retrival_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        
        instruction = "Please select the paper that is more likely to be cited by paper A from candidate papers."
            
        prompt_input = ""
        prompt_input = prompt_input + "Title of the Paper A: " + data_point['s_title'] + "\n"
        prompt_input = prompt_input + "Abstract of the Paper A: " + data_point['s_abs'] + "\n"
        prompt_input = prompt_input + "candidate papers: " + "\n"
        for i in range(len(data_point['nei_titles'])):
            prompt_input = prompt_input + str(i) + '. ' + data_point['nei_titles'][i] + "\n"
        
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res

    def _generate_sentence_prompt(self, data_point, eos_token):
        instruction = "Please generate the citation sentence of how Paper A cites paper B in its related work section."
            
        prompt_input = ""
        prompt_input = prompt_input + "Title of Paper A: " + (data_point['s_title'] if data_point['s_title'] != None else 'Unknown') + "\n"
        #prompt_input = prompt_input + "Abstract of Paper A: " + (data_point['s_abs'] if data_point['s_abs'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Title of Paper B: " + (data_point['t_title'] if data_point['t_title'] != None else 'Unknown') + "\n"
        #prompt_input = prompt_input + "Abstract of Paper B: " + (data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown') + "\n"

        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)

        return res

    def _generate_topic_prompt(self, data_point, eos_token):
        prompt_input = ""
        prompt_input = prompt_input + "Here are the information of a paper: \n"
        prompt_input = prompt_input + "Title of Paper: " + data_point['s_title'] + '\n' if data_point['s_title'] != None else 'Unknown' + "\n"
        prompt_input = prompt_input + "Abstract of Paper: " + data_point['s_abs'] + '\n' if data_point['s_abs'] != None else 'Unknown' + "\n"
        prompt_input = prompt_input + "Directlty give me the topics you select. \n"
            
        instruction = "I need to write the related work section for this paper. Could you suggest three most relevant topics to discuss in the related work section?\n"
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        print(res) 
        return res

    def _generate_paragraph_prompt(self, data_point, eos_token):
        instruction = "Please write a paragraph that review the research relationships between this paper and other cited papers."
        prompt_input = ""
        prompt_input = prompt_input + "Title of the paper: " + data_point['s_title'] + "\n"
        prompt_input = prompt_input + "Abstract of the paper: " + data_point['s_abs'] + "\n"
        prompt_input = prompt_input + "Topic of this paragraph: " + data_point['topic'] + "\n"
        prompt_input = prompt_input + "papers that should be cited in paragraph: \n"

        i = data_point['paper_citation_indicator']
        for paper_idx in range(len(data_point['nei_title'])):
            prompt_input = prompt_input + "[" + str(i) + "]. " + data_point['nei_title'][paper_idx]  + '.' + " Citation sentence of this paper in the paragraph: " + data_point['nei_sentence'][paper_idx] + '\n'
            i += 1
        
        prompt_input = prompt_input + "All the above cited papers should be included and each cited paper should be indicated with its index number. Note that you should not include the title of any paper\n"
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res

    def _generate_summary_prompt(self, data_point, eos_token):
        instruction = "Please combine the following paragraphs in a cohenrent way that also keeps the citations and make the flow between paragraphs more smoothly"
        instruction = "Add a sentence at the beginning of each paragraph to clarify its connection to the previous ones."
        
        prompt_input = ""
        prompt_input = prompt_input + "Title of the Paper: " + data_point['s_title'] + "\n"
        prompt_input = prompt_input + "Abstract of the Paper: " + data_point['s_abs'] + "\n"
        prompt_input = prompt_input + "Paragraphs that should be combined: " + "\n"
        
        i = 1
        for para in data_point['paragraphs']:
            prompt_input = prompt_input + " Paragraph " + str(i) + ": " + para + '\n'
            i += 1
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        return res

    def get_llm_response(self, prompt, model_type):
        if model_type == 'zeroshot':
            #generation_model.set_adapter('text')
            with self.generation_model.disable_adapter():
                pipe = pipeline(
                            "text-generation",
                            model=self.generation_model,
                            tokenizer=self.generation_tokenizer, 
                            #max_length=9096,
                            temperature=0.7,
                            top_p=0.95,
                            repetition_penalty=1.15,
                            max_new_tokens=2048,
                        )
                raw_output = pipe(prompt)
        
        if model_type == 'zeroshot_short':
            #generation_model.set_adapter('text')
            with self.generation_model.disable_adapter():
                pipe = pipeline(
                            "text-generation",
                            model=self.generation_model,
                            tokenizer=self.generation_tokenizer, 
                            #max_length=9096,
                            temperature=0.7,
                            top_p=0.95,
                            repetition_penalty=1.15,
                            max_new_tokens=256,
                        )
                raw_output = pipe(prompt)
        if model_type == 'instruction':
            self.generation_model.set_adapter('instruction')
            pipe = pipeline(
                        "text-generation",
                        model=self.generation_model,
                        tokenizer=self.generation_tokenizer, 
                        #max_length=9096,
                        temperature=0.7,
                        top_p=0.95,
                        repetition_penalty=1.15,
                        max_new_tokens=256,
                    )
            raw_output = pipe(prompt)
        return raw_output

    def single_paper_sentence_test(self, s_title, s_abs, t_title, t_abs):
        datapoint = {'s_title':s_title, 's_abs':s_abs, 't_title':t_title, 't_abs':t_abs}
        prompt = self._generate_sentence_prompt(datapoint, '')
        ans = self.get_llm_response(prompt, 'instruction')[0]['generated_text'] # instruction
        res = ans.strip().split(self.human_instruction[1])[-1]
        return res

    def single_paper_retrieval_test(self, s_title, s_abs, candidates):
        datapoint = {'s_title':s_title, 's_abs':s_abs, 'nei_titles':list(candidates), 't_title': ''}
        prompt = self._generate_retrival_prompt(datapoint, '')
        ans = self.get_llm_response(prompt, 'instruction')[0]['generated_text']
        print(prompt)
        res = ans.strip().split(self.human_instruction[1])[-1]
        print(res)
        return res

    def single_paper_topic_test(self, s_title, s_abs):
        datapoint = {'s_title':s_title, 's_abs':s_abs}
        prompt = self._generate_topic_prompt(datapoint, '')
        ans = self.get_llm_response(prompt, 'zeroshot_short')[0]['generated_text']
        res = ans.strip().split(self.human_instruction[1])[-1]
        return res

    def retrival_for_one_query(self, model, tokenizer, graph_data, id_2_raw_id_dict, id_2_title_abs, prompt):
        print("generate embeddings for all candidate nodes")
        try:
            raw_graph = graph_data
            all_candidate_embs_L1 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 1_emb.parquet')['embedding'].tolist())).half()
            all_candidate_embs_L2 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 2_emb.parquet')['embedding'].tolist())).half()
            all_candidate_embs_L3 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 3_emb.parquet')['embedding'].tolist())).half()
            all_query_embs = all_candidate_embs_L1 + all_candidate_embs_L2 + all_candidate_embs_L3
        except:
            all_query_embs = []
            pretrained_model_name = self.pretrained_model
            config = AutoConfig.from_pretrained(pretrained_model_name)
            hidden_size = config.hidden_size
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
            LLM_model.eval()

            tmp_list = list(id_2_title_abs.keys())
            i = 0
            in_batch_size=200
            pbar = tqdm(total=len(tmp_list))
            while i < len(tmp_list):
                paper_batch = tmp_list[i:i+in_batch_size]
                paper_text_batch = []
                for paper_id in paper_batch:
                    prompt = id_2_title_abs[paper_id][0] + id_2_title_abs[paper_id][1]
                    paper_text_batch.append(prompt)
                encoded_input = tokenizer(paper_text_batch, padding = True, truncation=True, max_length=512 , return_tensors='pt')
                
                with torch.no_grad():
                    output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
                    sentence_embedding = output[:, 0, :].cpu()
                all_query_embs.append(sentence_embedding)

                i += len(sentence_embedding)
                pbar.update(len(sentence_embedding))
            
            all_query_embs = torch.cat(all_query_embs, 0)
            pickle.dump(all_query_embs, open('llm_embeddings_'+self.pretrained_model.replace('/','_')+'_'+self.graph_name+'.pkl', 'wb'))
        
        Node_dataset = NodeDataset_QA(raw_graph, id_2_title_abs, self.neigh_num, tokenizer)
        Node_dataset.init_bert_embeddings(all_query_embs)

        loader = DataLoader(Node_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        all_node_embeddings = []
        all_nodes = []
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            batch_input = dict()
            for key in sample_batched.keys():
                batch_input[key] = sample_batched[key].float().cuda()      
            subnode_embeddings = model.generate_candidate_emb(batch_input).detach()
            all_node_embeddings.append(subnode_embeddings)
            all_nodes.append(batch_input['target'].detach().cpu())
        all_node_embeddings = torch.cat(all_node_embeddings, 0)
        all_nodes = torch.cat(all_nodes, 0).squeeze(-1)
        print(all_node_embeddings.size())
        print(all_node_embeddings[0][:10])

        # get test query embeddings
        print("generate embeddings for test paper")
        test_query_embs = {}
        print('---------------------------')
        pretrained_model_name = self.pretrained_model
        config = AutoConfig.from_pretrained(pretrained_model_name)
        hidden_size = config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
        LLM_model.eval()

        encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
        with torch.no_grad():
            output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
            sentence_embedding = output[:, 0, :]

        test_query_embs[0] = sentence_embedding[0].cpu()
        
        # init test dataset
        Test_dataset = TestEdgeDataset_QA(tokenizer, test_query_embs)
        print("test set eval")

        MRR_list, H1_list, H3_list, H10_list = [], [], [], []
        Top_10_list, Top_5_list = [], []
        all_querys, all_pred, all_truth = [], [], []
        loader = DataLoader(Test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            batch_input = dict()
            for key in sample_batched.keys():
                if 'query' in key:
                    batch_input[key] = sample_batched[key].float().cuda()
                else:
                    batch_input[key] = sample_batched[key].long().cuda()
            query_embeddings = model.generate_query_emb(batch_input)
            all_truth = self.get_result(batch_input['s'].cpu(), query_embeddings.detach(), all_node_embeddings, model, None, all_nodes, id_2_raw_id_dict) 
        return [id_2_title_abs[i][0] for i in all_truth] 

    def get_result(self, s_ind, query_embs, all_node_embeddings, model, truth, all_nodes, id_2_raw_id_dict):
        # get score of candidate samples
        R_list, RR_list, H1_list, H3_list, H10_list = [], [], [], [], []
        top_10_result, top_5_result = [], []
        all_nodes = np.array(all_nodes)
        query_ind_list = []
        top10_list = []
        candidate_list = []
        query_ind = s_ind[0]
        s_embeddings = query_embs[0]
        tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()
        sorted_scores, idxs = torch.sort(tmp_scores, descending = True)
        top_30 = all_nodes[[int(k) for k in idxs[:30]]]
        top_20 = all_nodes[[int(k) for k in idxs[:20]]]
        top_10 = all_nodes[[int(k) for k in idxs[:10]]]
        top_5 = all_nodes[[int(k) for k in idxs[:5]]]
        
        return top_10

    def single_paper_related_work_generation(self, s_title, s_abs):    
        
        citation_papers = []
        nei_sentence = []

        # summary query
        retrieval_query = self.single_paper_topic_test(s_title, s_abs)
        print("title: " + s_title)
        print("abs: " + s_abs)
        print(retrieval_query)

        topic_num = 3
        try:
            split_topics = retrieval_query.strip().split('\n')[1:]
            if split_topics[0] == '':
                split_topics = split_topics[1:]
            split_topics = split_topics[:topic_num]
        except:
            split_topics = retrieval_query.strip().split(':')[1]
            split_topics = split_topics.strip().split(';')
        split_topics = split_topics[:topic_num]
        print(split_topics)
        if len(split_topics) > topic_num:
            return ["too many topics", split_topics]

        for retrieval_query in split_topics: # each topic has 5 desired cited papers
            # retrieve papers
            candidate_citation_papers = self.retrival_for_one_query(self.retrieval_model, self.retrieval_tokenizer, self.whole_graph_data, self.whole_graph_id_2_raw_id_dict, self.whole_graph_id_2_title_abs, retrieval_query)
            print("candidate_citation_papers: " + str(candidate_citation_papers))

            topic_specific_citation_papers = []
            # filter papers
            for i in range(5):
                selected_paper = self.single_paper_retrieval_test(s_title, s_abs, candidate_citation_papers).replace(' \n','').replace('\n','')
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
            
            print("citation_papers: " + str(citation_papers))

        # generate citation sentence
        for topic_idx in range(len(citation_papers)):
            topic_specific_nei_sentence = []
            for paper_idx in range(len(citation_papers[topic_idx])):
                sentence = self.single_paper_sentence_test(s_title, s_abs, citation_papers[topic_idx][paper_idx], "")
                print(sentence)
                sentence = re.sub(r'\\(\S*)+\}', "", sentence)
                print(sentence)
                sentence = re.sub(r'\[(\S*)+\]', "", sentence)
                print(sentence)
                topic_specific_nei_sentence.append(sentence)
            nei_sentence.append(topic_specific_nei_sentence)
        
        print("nei_sentence: " + str(nei_sentence))

        # get each paragraph
        paragraphs = []
        paper_citation_indicator = 1
        for topic_idx in range(len(citation_papers)):
            datapoint = {'s_title': s_title, 
                        's_abs': s_abs, 
                        'nei_title': citation_papers[topic_idx], 
                        'nei_sentence': nei_sentence[topic_idx], 
                        'topic': split_topics[topic_idx], 
                        'paper_citation_indicator': paper_citation_indicator}
            
            prompt = self._generate_paragraph_prompt(datapoint, self.generation_tokenizer.eos_token)
            ans = self.get_llm_response(prompt, 'zeroshot')[0]['generated_text']
            res = ans.strip().split(self.human_instruction[1])[-1]
            paragraphs.append(res)

            paper_citation_indicator = paper_citation_indicator + len(nei_sentence[topic_idx])

        # summary
        datapoint = {'s_title': s_title, 's_abs': s_abs, 'paragraphs': paragraphs}
        prompt = self._generate_summary_prompt(datapoint, self.generation_tokenizer.eos_token)
        ans = self.get_llm_response(prompt, 'zeroshot')[0]['generated_text']
        res = ans.strip().split(self.human_instruction[1])[-1]

        return [s_title, s_abs, split_topics, citation_papers, nei_sentence, prompt, res]


def gen_related_work():
    LitFM_example = LitFM()
    sample_title = "LitFM: A Retrieval Augmented Structure-aware Foundation Model For Citation Graphs"
    sample_abs = "With the advent of large language models (LLMs), managing scientific literature via LLMs has become a promising direction of research. However, existing approaches often overlook the rich structural and semantic relevance among scientific literature, limiting their ability to discern the relationships between pieces of scientific knowledge, and suffer from various types of hallucinations. These methods also focus narrowly on individual downstream tasks, limiting their applicability across use cases. Here we propose LitFM, the first literature foundation model designed for a wide variety of practical downstream tasks on domain-specific literature, with a focus on citation information. At its core, LitFM contains a novel graph retriever to integrate graph structure by navigating citation graphs and extracting relevant literature, thereby enhancing model reliability. LitFM also leverages a knowledge-infused LLM, fine-tuned through a welldeveloped instruction paradigm. It enables LitFM to extract domain-specific knowledge from literature and reason relationships among them. By integrating citation graphs during both training and inference, LitFM can generalize to unseen papers and accurately assess their relevance within existing literature. Additionally, we introduce new large-scale literature citation benchmark datasets on three academic fields, featuring sentence-level citation information and local context. "
    related_work = LitFM_example.single_paper_related_work_generation(sample_title, sample_abs)
    print(related_work)

    # whole_graph_data_raw = nx.read_gexf("datasets/quantum_graph.gexf", node_type=None, relabel=False, version='1.2draft')
    # sample_list = []
    # for paper_id in whole_graph_data_raw.nodes():
    #     title = whole_graph_data_raw.nodes()[paper_id]['title']
    #     abstract = whole_graph_data_raw.nodes()[paper_id]['abstract']
    #     sample_list.append([title, abstract])
    #     if len(sample_list)>= 20:
    #         break
    # for i in range(len(sample_list)):
    #     related_work_file = open("llama_related_work_results_quantum.txt", "a")
    #     result_list = LitFM_example.single_paper_related_work_generation(sample_list[i][0], sample_list[i][1])
    #     print(result_list)
    #     related_work_file.write(str(result_list)+'\n')
    #     related_work_file.close()


if __name__ == "__main__":
    gen_related_work()

# CUDA_VISIBLE_DEVICES=0 python inference_benchmark_COT_function.py