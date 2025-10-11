import argparse
import torch
import json
import random
import networkx as nx
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from bert_score import score
from tqdm import tqdm
import os
import yaml


"""
Ad-hoc sanity check to see if model outputs something coherent
Not a robust inference platform!
"""


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")


def get_bert_score(candidate, reference):
    P, R, F1 = score([candidate], [reference],lang="en")
    return P, R, F1

def _generate_LP_prompt(data_point: dict):
    instruction = "Determine if paper A will cite paper B.\n"

    prompt_input = ""
    prompt_input = prompt_input + "Title of Paper A: " + (data_point['s_title'] if data_point['s_title'] != None else 'Unknown') + "\n"
    prompt_input = prompt_input + "Abstract of Paper A: " + (data_point['s_abs'] if data_point['s_abs'] != None else 'Unknown') + "\n"
    prompt_input = prompt_input + "Title of Paper B: " + (data_point['t_title'] if data_point['t_title'] != None else 'Unknown') + "\n"
    prompt_input = prompt_input + "Abstract of Paper B: " + (data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown') + "\n"

    if tokenizer.chat_template is None:
        res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    else:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": template["prompt_input"].format(instruction=instruction, input=prompt_input)}
        ]

    return res

def _generate_paper_retrieval_prompt(data_point: dict):
    instruction = "Please select the paper that is more likely to be cited by paper A from candidate papers.\n"

    prompt_input = ""
    prompt_input = prompt_input + "Title of the Paper A: " + data_point['s_title'] + "\n"
    prompt_input = prompt_input + "Abstract of the Paper A: " + data_point['s_abs'] + "\n"
    prompt_input = prompt_input + "candidate papers: " + "\n"
    for i in range(len(data_point['nei_titles'])):
        prompt_input = prompt_input + str(i) + '. ' + data_point['nei_titles'][i] + "\n"
    
    if tokenizer.chat_template is None:
        res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    else:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": template["prompt_input"].format(instruction=instruction, input=prompt_input)}
        ]

    return res

def _generate_abstract_2_title_prompt(data_point: dict):
    instruction = "Please generate the title of paper based on its abstract.\n"

    prompt_input = ""
    prompt_input = prompt_input + "Abstract: " + data_point['abs'] + "\n"

    if tokenizer.chat_template is None:
        res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    else:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": template["prompt_input"].format(instruction=instruction, input=prompt_input)}
        ]

    return res

def _generate_citation_sentence_prompt(data_point: dict):
    instruction = "Please generate the citation sentence of how Paper A cites paper B in its related work section.\n"
    
    prompt_input = ""
    prompt_input = prompt_input + "Title of Paper A: " + (data_point['s_title'] if data_point['s_title'] != None else 'Unknown') + "\n"
    prompt_input = prompt_input + "Abstract of Paper A: " + (data_point['s_abs'] if data_point['s_abs'] != None else 'Unknown') + "\n"
    prompt_input = prompt_input + "Title of Paper B: " + (data_point['t_title'] if data_point['t_title'] != None else 'Unknown') + "\n"
    prompt_input = prompt_input + "Abstract of Paper B: " + (data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown') + "\n"

    if tokenizer.chat_template is None:
        res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    else:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": template["prompt_input"].format(instruction=instruction, input=prompt_input)}
        ]

    return res

def _generate_abstract_completion_prompt(data_point: dict):
    instruction = "Please complete the abstract of a paper.\n"

    prompt_input = ""
    prompt_input = prompt_input + "Title: " + data_point['title'] if data_point['title'] != None else 'Unknown' + "\n"
    
    split_abs = data_point['abs'][: int(0.3*len(data_point['abs']))]
    prompt_input = prompt_input + "Part of abstract: " + split_abs + "\n"

    if tokenizer.chat_template is None:
        res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    else:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": template["prompt_input"].format(instruction=instruction, input=prompt_input)}
        ]

    return res

def _generate_intro_2_abstract_prompt(data_point: dict, context_window):
    instruction = "Please generate the abstract of paper based on its introduction section."

    prompt_input = ""
    prompt_input = prompt_input + "Introduction: " + data_point['intro'] + "\n"

    # Reduce it to make it fit
    prompt_input = prompt_input[:int(context_window*2)]

    if tokenizer.chat_template is None:
        res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    else:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": template["prompt_input"].format(instruction=instruction, input=prompt_input)}
        ]

    return res


def get_llm_response(prompt, task):
    if task == 'sentence':
        return pipe_sentence(prompt)
    if task == 'LP':
        return pipe_LP(prompt)
    if task == 'abstract':
        return pipe_abstract(prompt)
    if task == 'title':
        return pipe_title(prompt)
    if task == 'retrieval':
        return pipe_retrieval(prompt)
    if task == 'intro':
        return pipe_intro(prompt)


def test_sentence():
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []

    result_dict = {}
    # pos test
    for i in tqdm(range(len(test_data))):
        source, target = test_data[i][0], test_data[i][1]
        source_title, source_abs = raw_id_2_title_abs[source]
        target_title, target_abs = raw_id_2_title_abs[target]
        
        s_nei = list(nx.all_neighbors(raw_graph, source))
        s_nei_list = list(set(s_nei) - set([source]) - set([target]))[:10]
        s_nei_titles = [raw_id_2_title_abs[i][0] for i in s_nei_list]

        t_nei = list(nx.all_neighbors(raw_graph, target))
        t_nei_list = list(set(t_nei) - set([source]) - set([target]))[:10]
        t_nei_titles = [raw_id_2_title_abs[i][0] for i in t_nei_list]

        t_nei_sentence = []
        for i in range(len(t_nei_list)):
            tmp_sentence = raw_id_pair_2_sentence[(t_nei_list[i], target)] if (t_nei_list[i], target) in raw_id_pair_2_sentence.keys() else ''
            if len(tmp_sentence) != 0:
                t_nei_sentence.append(tmp_sentence)

        citation_sentence = raw_id_pair_2_sentence[(source, target)] if (source, target) in raw_id_pair_2_sentence.keys() else raw_id_pair_2_sentence[(target, source)]
        
        datapoint = {
            's_title':source_title,
            's_abs':source_abs,
            't_title':target_title,
            't_abs':target_abs,
            's_nei':s_nei_titles,
            't_nei':t_nei_titles,
            't_nei_sentence':t_nei_sentence,
            'sentence': citation_sentence,
        }

        prompt = _generate_citation_sentence_prompt(datapoint)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'sentence')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'sentence')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1])[-1]

        result_dict[(source, target)] = [source_title, source_abs, target_title, target_abs, citation_sentence, res]
        Bert_p, Bert_r, Bert_f = get_bert_score(res, citation_sentence)

        if verbose:
            print("Answer is:", ans)
            print("Stripped result is:", res)
            print("Citation sentence:", citation_sentence)

        Bert_p_list.append(Bert_p.item())
        Bert_r_list.append(Bert_r.item())
        Bert_f_list.append(Bert_f.item())

        if verbose:
            print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])

    return np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)


def test_LP():
    result_list = []
    # pos test
    for i in tqdm(range(len(test_data))):
        source, target = test_data[i][0], test_data[i][1]
        source_title, source_abs = raw_id_2_title_abs[source]
        target_title, target_abs = raw_id_2_title_abs[target]

        datapoint = {
            's_title':source_title,
            's_abs':source_abs,
            't_title':target_title,
            't_abs':target_abs,
            'label':'yes',
        }

        prompt = _generate_LP_prompt(datapoint)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'LP')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'LP')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1])[-1]

        if verbose:
            print("Answer is:", res)

        if 'yes' in res[:4] or 'Yes' in res[:4]:
            result_list.append(1)
        else:
            result_list.append(0)

        if verbose:
            print("Current value:", np.mean(result_list))


     # neg test
    for i in tqdm(range(len(test_data))):
        source, target = test_data[i][0], random.sample(list(graph_data.nodes()), 1)[0]
        source_title, source_abs = raw_id_2_title_abs[source]
        target_title, target_abs = raw_id_2_title_abs[target]

        datapoint = {
            's_title':source_title,
            's_abs':source_abs,
            't_title':target_title,
            't_abs':target_abs,
            'label':'no',
        }

        prompt = _generate_LP_prompt(datapoint)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'LP')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'LP')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1])[-1]

        if verbose:
            print("Answer is:", res)

        if 'No' in res[:4] or 'no' in res[:4]:
            result_list.append(1)
        else:
            result_list.append(0)

        if verbose:
            print("Current value:", np.mean(result_list))

    return np.mean(result_list)


def test_title_generate():
    result_dict = {}
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []
    # pos test
    for i in tqdm(range(len(test_data))):
        source, _ = test_data[i][0], test_data[i][1]
        title, abstract = raw_id_2_title_abs[source]
        if title == None or abstract == None:
            continue

        datapoint = {'title':title, 'abs':abstract}

        prompt = _generate_abstract_2_title_prompt(datapoint)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'title')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'title')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1])[-1]

        result_dict[source] = [title, abstract, res]

        if verbose:
            print(ans)
            print(res)
            print(title)

        Bert_p, Bert_r, Bert_f = get_bert_score(res, title)
        Bert_p_list.append(Bert_p.item())
        Bert_r_list.append(Bert_r.item())
        Bert_f_list.append(Bert_f.item())

        if verbose:
            print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])

    return np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)


def test_abs_completion():
    result_dict = {}
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []
    # pos test
    for i in tqdm(range(len(test_data))):
        source, _ = test_data[i][0], test_data[i][1]
        title, abstract = raw_id_2_title_abs[source]
        if title == None or abstract == None:
            continue

        datapoint = {'title':title, 'abs':abstract}

        prompt = _generate_abstract_completion_prompt(datapoint)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'abstract')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'abstract')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1])[-1]

        result_dict[source] = [title, abstract, res]

        if verbose:
            print(ans)
            print(res)
            print(abstract)

        Bert_p, Bert_r, Bert_f = get_bert_score(res, abstract)

        Bert_p_list.append(Bert_p.item())
        Bert_r_list.append(Bert_r.item())
        Bert_f_list.append(Bert_f.item())

        if verbose:
            print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])

    return np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)


def test_retrieval():
    result_list = []
    # pos test
    for i in tqdm(range(len(test_data))):
        source, target = test_data[i][0], test_data[i][1]
        source_title, source_abs = raw_id_2_title_abs[source]
        target_title, _ = raw_id_2_title_abs[target]

        neighbors = list(nx.all_neighbors(raw_graph, source))
        sample_node_list = list(set(raw_graph.nodes()) - set(neighbors) - set([source]) - set([target]))
        sampled_neg_nodes = random.sample(sample_node_list, 5) + [target]
        random.shuffle(sampled_neg_nodes)

        datapoint = {
            's_title':source_title,
            's_abs':source_abs,
            't_title':target_title,
            'nei_titles':[raw_id_2_title_abs[node][0] for node in sampled_neg_nodes]
        }

        prompt = _generate_paper_retrieval_prompt(datapoint)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'retrieval')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'retrieval')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1])[-1].lower()
        target_title = target_title.lower()

        if verbose:
            print(ans)
            print("###GT: " + target_title)
            print(res)

        if target_title in res or res in target_title:
            result_list.append(1)
        else:
            result_list.append(0)

        if verbose:
            print([sum(result_list), len(result_list)])

    if verbose:
        print([sum(result_list), len(result_list)])

    return np.mean(result_list)

def test_intro_2_abs():
    result_dict = {}
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []
    # pos test
    for i in tqdm(range(len(test_data))):
        source, target = test_data[i][0], test_data[i][1]

        if source not in raw_id_2_intro:
            source = target

        if source not in raw_id_2_intro:
            continue

        title, abstract = raw_id_2_title_abs[source]
        intro = raw_id_2_intro[source]

        datapoint = {'abs':abstract, 'intro':intro}

        prompt = _generate_intro_2_abstract_prompt(datapoint, tokenizer.model_max_length)
        if tokenizer.chat_template is None:
            ans = get_llm_response(prompt, 'intro')[0]['generated_text']
        else:
            ans = get_llm_response(prompt, 'intro')[0]['generated_text'][-1]['content']

        res = ans.strip().split(human_instruction[1]+'\n')[-1]

        result_dict[source] = [title, abstract, res]

        if verbose:
            print(ans)
            print(res)
            print(abstract)

        Bert_p, Bert_r, Bert_f = get_bert_score(res, abstract)

        Bert_p_list.append(Bert_p.item())
        Bert_r_list.append(Bert_r.item())
        Bert_f_list.append(Bert_f.item())

        if verbose:
            print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])

    return np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", help="Path to the config YAML file")
    parser.add_argument("-adapter_path", help="Path to the config YAML file")
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)
    random.seed(42)
    print("Load model")
    model_path = config["eval"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = 2048
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, load_in_8bit=True)
    if base_model.device.type != 'cuda':
        base_model.to('cuda')

    adapter_save_path = args.adapter_path
    model = PeftModel.from_pretrained(base_model, adapter_save_path)
    model = model.merge_and_unload()

    pipe_LP = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    pipe_title = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    pipe_abstract = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    pipe_intro = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    pipe_sentence = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    pipe_retrieval = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    graph_path = config["eval"]["graph_path"]
    graph_data = nx.read_gexf(graph_path, node_type=None, relabel=False, version='1.2draft')

    raw_graph = graph_data

    test_set_size = 50
    all_test_nodes = set(list(graph_data.nodes())[:test_set_size])
    all_train_nodes = set(list(graph_data.nodes())[test_set_size:])

    raw_id_2_title_abs = dict()
    for paper_id in list(graph_data.nodes()):
        title = graph_data.nodes()[paper_id]['title']
        abstract = graph_data.nodes()[paper_id]['abstract']
        raw_id_2_title_abs[paper_id] = [title, abstract]

    raw_id_pair_2_sentence = dict()
    for edge in list(graph_data.edges()):
        sentence = graph_data.edges()[edge].get('sentence', '')
        raw_id_pair_2_sentence[edge] = sentence

    raw_id_2_intro = dict()
    for paper_id in list(graph_data.nodes())[test_set_size:]:
        if graph_data.nodes[paper_id]['introduction'] != '':
            intro = graph_data.nodes[paper_id]['introduction']
            raw_id_2_intro[paper_id] = intro

    test_data = []
    edge_list = []
    for edge in list(raw_graph.edges()):
        src, tar = edge
        if src not in all_test_nodes and tar not in all_test_nodes:
            edge_list.append(edge)
        else:
            test_data.append(edge)


    with open('configs/alpaca.json') as fp:
        template = json.load(fp)
    human_instruction = ['### Input:', '### Response:']

    verbose = config["eval"]["verbose"]

    LP_score = test_LP()
    retrieval_score = test_retrieval()
    title_p, title_r, title_f = test_title_generate()
    sentence_p, sentence_r, sentence_f = test_sentence()
    abstract_p, abstract_r, abstract_f = test_abs_completion()
    intro_p, intro_r, intro_f = test_intro_2_abs()

    print("Retrieval Score:", retrieval_score)
    print("LP Score:", LP_score)
    print("Title:", [title_p, title_r, title_f])
    print("Sentence:", [sentence_p, sentence_r, sentence_f])
    print("Abstract:", [abstract_p, abstract_r, abstract_f])
    print("Intro:", [intro_p, intro_r, intro_f])

    results = {
        "LP_score": LP_score,
        "retrieval_score": retrieval_score,
        "title": {
            "precision": title_p,
            "recall": title_r,
            "f1": title_f
        },
        "sentence": {
            "precision": sentence_p,
            "recall": sentence_r,
            "f1": sentence_f
        },
        "abstract": {
            "precision": abstract_p,
            "recall": abstract_r,
            "f1": abstract_f
        },
        "intro": {
            "precision": intro_p,
            "recall": intro_r,
            "f1": intro_f
        },
    }

    graph_name = graph_path.split('/')[-1].split('.')[0]

    try:
        os.mkdir("eval")
    except:
        pass

    with open(f"eval/{graph_name}_results.json", "w") as f:
        json.dump(results, f)
