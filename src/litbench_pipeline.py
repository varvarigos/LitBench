from Lora_finetune_benchmark import *
from utils.utils import *
from utils.graph_utils import *
from utils.gradio_utils import *
from retriever.bert_retriever import retriever
from tasks.abs_2_title import abs_2_title
from tasks.abs_completion import abs_completion
from tasks.citation_sentence import citation_sentence
from tasks.intro_2_abs import intro_2_abs
from tasks.link_pred import link_pred
from tasks.paper_retrieval import paper_retrieval
from tasks.influential_papers import influential_papers
from tasks.gen_related_work import gen_related_work
import random
import json
import os
import re
import networkx as nx
import tarfile
import gzip
import time
import urllib.request
from tqdm import tqdm
from colorama import Fore
import wandb
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
import signal
import gzip
import time
import torch
from peft.peft_model import PeftModel
from datasets import load_dataset



# Function to determine the chatbot's first message based on user choices
def setup(download_option, train_option):
    download_papers.value = (download_option == "Download Paper")
    train_model.value = (train_option == "Train")

    if train_option == "Train":
        initial_message = [{"role": "assistant", "content": "Hello, what domain are you interested in?"}]
    else:
        initial_message = [{"role": "assistant", "content": "Please provide your task prompt."}]

    return gr.update(visible=False), gr.update(visible=True), f"Download: {download_option}\nTrain: {train_option}", initial_message


# Function to toggle the selected task based on user input
def update_button_styles(selected_task):
    """Update button styles based on selection."""
    return [gr.update(variant="primary" if selected_task == prompt else "secondary") for prompt in task_list]


# Fetch and store arXiv source files
def fetch_arxiv_papers(papers_to_download):
    # Download the arXiv metadata file if it doesn't exist
    dataset = 'datasets/arxiv-metadata-oai-snapshot.json'
    data = []
    if not os.path.exists(dataset):
        os.system("wget https://huggingface.co/spaces/ddiddu/simsearch/resolve/main/arxiv-metadata-oai-snapshot.json -P ./datasets")

    with open(dataset, 'r') as f:
        for line in f: 
            data.append(json.loads(line))

    papers = [d for d in data]
    paper_ids = [d['id'] for d in data]
    paper_titles = [
        (
            re.sub(r' +', ' ', re.sub(r'[\n]+', ' ', paper['title']))
            .replace("\\emph", "")
            .replace("\\emp", "")
            .replace("\\em", "")
            .replace(",", "")
            .replace("{", "")
            .replace("}", "")
            .strip(".")
            .strip()
            .strip(".")
            .lower()
        )
        for paper in papers
    ]
    paper_dict = {
        k:v
        for k,v in zip(paper_titles, paper_ids)
    }


    total_papers = len(papers_to_download)
    download_progress_bar=gr.Progress()
    
    llm_resp = []
    results = {
        "Number of papers": 0,
        "Number of latex papers": 0,
        "Number of bib files": 0,
        "Number of bbl files": 0,
        "Number of inline files": 0,
        "Number of introductions found": 0,
        "Number of related works found": 0,
        "Number of succesful finding of extracts": 0
    }
    num_papers, num_edges, t, iter_ind = 0, 0, 0, 0
    graph = {}

    for paper_name in tqdm(papers_to_download):
        results["Number of papers"] += 1
        print(
            Fore.BLUE + "Number of papers processed: {} \n Number of edges found: {} \n Time of previous iter: {} \n Now processing paper: {} \n\n"
            .format(num_papers, num_edges, time.time()-t, paper_name) + Fore.RESET
        )
        t = time.time()
        num_papers += 1

        # Prepare the paper name for downloading and saving
        paper_name_download = paper_name
        if re.search(r'[a-zA-Z]', paper_name) is not None:
            paper_name = "".join(paper_name.split('/'))
        tar_file_path = save_zip_directory + paper_name + '.tar.gz'

        # Attempt to download the paper source files from arXiv
        try:
            # Track start time for download
            t1 = time.time()
            urllib.request.urlretrieve(
            "https://arxiv.org/src/"+paper_name_download,
            tar_file_path)
        except Exception as e:
            print("Couldn't download paper {}".format(paper_name))
            # Skip to the next paper if download fails
            continue

        # Define the directory where the paper will be extracted
        extracted_dir = save_directory + paper_name + '/'
        isExist = os.path.exists(extracted_dir)
        if not isExist:
            os.makedirs(extracted_dir)

        # Attempt to extract the tar.gz archive
        try:
            tar = tarfile.open(tar_file_path)
            tar.extractall(extracted_dir)
            tar.close()
        except Exception as e:
            # If tar extraction fails, attempt to read and extract using gzip
            try:
                with gzip.open(tar_file_path, 'rb') as f:
                    file_content = f.read()

                # Save the extracted content as a .tex file
                with open(extracted_dir+paper_name+'.tex', 'w') as f:
                    f.write(file_content.decode())
            except Exception as e:
                print("Could not extract paper id: {}".format(paper_name))
                # Skip this paper if extraction fails
                continue 

        try:
            # Perform initial cleaning and get the main TeX file
            initial_clean(extracted_dir, config=False)
            main_file = get_main(extracted_dir)

            # If no main TeX file is found, remove the downloaded archive and continue
            if main_file == None:
                print("No tex files found")
                os.remove(tar_file_path)
                continue

            # Check if the main TeX file contains a valid LaTeX document
            h = check_begin(main_file)
            if h == True:
                results["Number of latex papers"] += 1
                # Flag to check for internal bibliography
                check_internal = 0
                # Dictionary to store bibliographic references
                final_library = {}

                # Identify bibliography files (.bib or .bbl)
                bib_files = find_bib(extracted_dir)
                if bib_files == []:
                    bbl_files = find_bbl(extracted_dir)
                    if bbl_files == []:
                        # No external bibliography found
                        check_internal = 1
                    else:
                        final_library = get_library_bbl(bbl_files)
                        results["Number of bbl files"] += 1
                else:
                    results["Number of bib files"] += 1
                    final_library = get_library_bib(bib_files)

                # Apply post-processing to clean the TeX document
                main_file = post_processing(extracted_dir, main_file)

                # Read the cleaned LaTeX document content
                descr = main_file
                content = read_tex_file(descr)

                # If configured, store the raw content in the graph
                if config['processing']['keep_unstructured_content']:
                    graph[paper_name] = {'content': content}

                # Check for inline bibliography within the LaTeX document
                if check_internal == 1:
                    beginning_bib = '\\begin{thebibliography}'
                    end_bib = '\\end{thebibliography}'

                    if content.find(beginning_bib) != -1 and content.find(end_bib) != -1:
                        bibliography = content[content.find(beginning_bib):content.find(end_bib) + len(end_bib)]
                        save_bbl = os.path.join(extracted_dir, "bibliography.bbl")

                        results["Number of inline files"] += 1
                        with open(save_bbl, "w") as f:
                            f.write(bibliography)

                        final_library = get_library_bbl([save_bbl])

                # If no valid bibliography is found, skip processing citations
                if final_library == {}:
                    print("No library found...")
                    continue

                # Extract relevant sections such as "Related Work" and "Introduction"
                related_works = get_related_works(content)
                if related_works  != '':
                    graph[paper_name]['Related Work'] = related_works
                    results["Number of intro/related found"] += 1
                
                intro = get_intro(content)
                if intro  != '':
                    graph[paper_name]['Introduction'] = intro
                    results["Number of introductions found"] += 1

                # Extract citation sentences from the introduction and related works
                sentences_citing = get_citing_sentences(intro + '\n' + related_works)
                
                # Map citations to corresponding papers
                raw_sentences_citing = {}
                for k,v in sentences_citing.items():
                    new_values = []
                    for item in v:
                        try:
                            new_values.append(paper_dict[final_library[item]['title']])
                        except Exception as e:
                            pass
                    if new_values != []:
                        raw_sentences_citing[k] = new_values

                # Construct citation edges
                edges_set = []
                for k,v in raw_sentences_citing.items():
                    for item in v:
                        edges_set.append((paper_name_download, item, {"sentence":k}))

                iter_ind +=1
                if len(edges_set) !=0:
                    results["Number of succesful finding of extracts"] += 1
                    graph[paper_name]['Citations'] = edges_set
                    num_edges += len(edges_set)

                # Save progress after every 10 iterations
                if iter_ind % 10 == 0:
                    print("Saving graph now")
                    with open(save_path, 'w') as f:
                        json.dump(results, f)
                    with open(save_graph, 'w') as f:
                        json.dump(graph, f)
        
        except Exception as e:
            print("Could not get main paper {}".format(paper_name))

        # Update the progress bar after processing each paper
        download_progress_bar(num_papers / total_papers)
        

        # Ensure a minimum time gap of 3 seconds between iterations to avoid bans from arXiv
        t2 = time.time()  # End time
        elapsed_time = t2 - t1
        if elapsed_time < 3:
            time.sleep(3 - elapsed_time)


    # Final saving of processed data
    with open(save_graph, 'w') as f:
        json.dump(graph, f)
    with open(save_path, 'w') as f:
        json.dump(results, f)


    # Log final completion message
    llm_resp.append("✅ Successfully downloaded and cleaned {} papers.".format(results["Number of latex papers"]))
    return "\n".join(llm_resp)


# Chat prediction function
def predict(message, history, selected_task):
    global model
    # Initialize the conversation string
    conversation = ""

    # Parse the history: Gradio `type="messages"` uses dictionaries with 'role' and 'content'
    for item in history:
        if item["role"] == "assistant":
            conversation += f"<bot>: {item['content']}\n"
        elif item["role"] == "user":
            conversation += f"<human>: {item['content']}\n"

    # Add the user's current message to the conversation
    conversation += f"<human>: {message}\n<bot>:"

    # Handle preferences
    if len(history) == 0:
        if not download_papers.value and train_model.value:
            yield f"✅ Using predefined graph: {predef_graph}"
            G = nx.read_gexf(predef_graph)
        elif not download_papers.value and not train_model.value:
            yield "✅ Loading model from configuration file."

            adapter_path = config["directories"]["pretrained_model"]
            peft_model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16)

            # change the global model with peft model
            model = peft_model

        time.sleep(2.5)

    if not (len(history) == 0 and (train_model.value or download_papers.value)):
        # Streamer for generating responses
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        stop = StopOnTokens()

        generate_kwargs = {
            "streamer": streamer,
            "max_new_tokens": 1000,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "temperature": 0.7,
            "no_repeat_ngram_size": 2,
            "num_beams": 1,
            "stopping_criteria": StoppingCriteriaList([stop]),
        }

        def generate_response(model, generate_kwargs, selected_task):
            global out
            if selected_task == "Abstract Completion":
                prompt = abs_completion(message, template)
            elif selected_task == "Title Generation": 
                prompt = abs_2_title(message, template)
            elif selected_task == "Citation Recommendation":
                prompt = paper_retrieval(message, template)
            elif selected_task == "Citation Sentence Generation":
                prompt = citation_sentence(message, template)
            elif selected_task == "Citation Link Prediction":
                prompt = link_pred(message, template)
            elif selected_task == "Introduction to Abstract":
                prompt = intro_2_abs(message, template, tokenizer.model_max_length)
            elif selected_task == "Influential Papers Recommendation":
                if download_papers.value:
                    graph = nx.read_gexf(gexf_file)
                    out = influential_papers(message, graph)
                else:
                    graph = nx.read_gexf(predef_graph)
                    out = influential_papers(message, graph)
            elif selected_task == "Related Work Generation":
                out = gen_related_work()
            else:
                prompt = conversation + f"<human>: {message}\n<bot>:"

            if selected_task != "Influential Papers Recommendation":
                model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
                generate_kwargs["inputs"] = model_inputs["input_ids"]
                generate_kwargs["attention_mask"] = model_inputs["attention_mask"]

                response = model.generate(**generate_kwargs)
                streamer.put(response)


        # Generate the response in a separate thread
        t = Thread(target=generate_response,
                    kwargs={
                       "model": model,
                       "generate_kwargs": generate_kwargs,
                       "selected_task": selected_task
                    })

        global out
        out = None
        t.start()
        
        # Stream the partial response
        if selected_task != "Influential Papers Recommendation":
            partial_message = ""
            for new_token in streamer:
                if new_token != '<':  # Ignore placeholder tokens
                    partial_message += new_token
                    yield partial_message
        else:
            while out == None:
                time.sleep(0.1)
            yield out

    # Fetch arXiv papers if the user opted to download them
    if len(history) == 0:
        if download_papers.value:
            # Fetch relevant papers
            yield "📥 Retrieving relevant papers..."

            retrieve_progress = gr.Progress()
            for percent in retriever(message, retrieval_nodes_path):
                retrieve_progress(percent)

            with open(retrieval_nodes_path, "r") as f:
                data_download = json.load(f)

            papers_to_download = list(data_download.keys())

            yield f"📥 Fetching {len(papers_to_download)} arXiv papers' source files... Please wait."

            content = fetch_arxiv_papers(papers_to_download)
            yield content
            time.sleep(2.5)
    

    # Train the model with the retrieved graph
    if len(history) == 0:
        if train_model.value:
            training_progress=gr.Progress()

            training_progress(0.0)

            # If the user opted to download papers, use the retrieved graph, else use the predefined graph
            if download_papers.value:
                yield "🚀 Training the model with the retrieved graph..."
                
                with open(save_graph, "r") as f:
                    data_graph = json.load(f)
                
                renamed_data = {
                    "/".join(re.match(r"([a-z-]+)([0-9]+)", key, re.I).groups()) if re.match(r"([a-z-]+)([0-9]+)", key, re.I) else key: value
                    for key, value in data_graph.items()
                }
                
                concept_data = load_dataset("json", data_files="datasets/arxiv_topics.jsonl")
                id2topics = {
                    entry["paper_id"]: [entry["Level 1"], entry["Level 2"], entry["Level 3"]]
                    for entry in concept_data["train"]
                }

                dataset = 'datasets/arxiv-metadata-oai-snapshot.json'
                data = []
                if not os.path.exists(dataset):
                    os.system("wget https://huggingface.co/spaces/ddiddu/simsearch/resolve/main/arxiv-metadata-oai-snapshot.json -P ./datasets")
                with open(dataset, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
                papers = {d['id']: d for d in data}

                G = nx.DiGraph()
                for k in renamed_data:
                    if k not in G and k in papers:
                        G.add_node(
                            k,
                            title=papers[k]['title'],
                            abstract=papers[k]['abstract'],
                            introduction=renamed_data[k].get('Introduction', '') if renamed_data[k].get('Introduction', '') != '\n' else '',
                            related=renamed_data[k].get('Related Work', '') if renamed_data[k].get('Related Work', '') != '\n' else '',
                            concepts=", ".join(list(set(item for sublist in id2topics[k] for item in sublist))) if k in id2topics else ''
                        )
                    if 'Citations' in renamed_data[k]:
                        for citation in renamed_data[k]['Citations']:
                            source, target, metadata = citation
                            sentence = metadata.get('sentence', '')  # Extract sentence or default to empty string

                            if target not in G and target in papers:
                                G.add_node(
                                    target,
                                    title=papers[target]['title'],
                                    abstract=papers[target]['abstract'],
                                    introduction=renamed_data[target].get('Introduction', '') if target in renamed_data and renamed_data[target].get('Introduction', '') != '\n'  else '',
                                    related=renamed_data[target].get('Related Work', '') if target in renamed_data and renamed_data[target].get('Related Work', '') != '\n'  else '',
                                    concepts=", ".join(list(set(item for sublist in concept_data[target].values() for item in sublist))) if target in concept_data else ''
                                )
                            
                            G.add_edge(source, target, sentence=sentence)

                G.remove_nodes_from(list(nx.isolates(G)))


            nx.write_gexf(G, gexf_file)
            print(f"Processed graph written to {gexf_file}")
        

            wandb.init(project='qlora_train')
            index = 1

            if download_papers.value:
                trainer = QloraTrainer_CS(config=config, index=index, use_predefined_graph=False)
            else:
                trainer = QloraTrainer_CS(config=config, index=index, use_predefined_graph=True)

            print("Load base model")
            trainer.load_base_model()


            print("Start training")
            def update_progress():
                # Wait for the trainer to be initialized
                while trainer.transformer_trainer is None:
                    time.sleep(0.5)

                # Update the progress bar until training is complete
                while trainer.transformer_trainer.state.global_step != trainer.transformer_trainer.state.max_steps:
                    progress_bar = (
                        trainer.transformer_trainer.state.global_step /
                        trainer.transformer_trainer.state.max_steps
                    )
                    training_progress(progress_bar)
                    time.sleep(0.5)
                training_progress(1.0)

            t1 = Thread(target=trainer.train)
            t1.start()
            t2 = Thread(target=update_progress())
            t2.start()
            t1.join()
            t2.join()

            yield "🎉 Model training complete! Please provide your task prompt."

            adapter_path = f"{config['model_output_dir']}/{config['model_name']}_{str(index)}_adapter_test_graph"
            peft_model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16)
            
            # change the global model with peft model
            model = peft_model



if __name__ == "__main__":
    print("This is running in a virtual environment: {}".format(is_venv()))

    config = read_yaml_file("conf/config.yaml")
    template_file_path = 'conf/alpaca.json'
    template = json.load(open(template_file_path, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"


    seed_no = config['processing']['random_seed']
    model_name = config['base_model']
    save_zip_directory = config['directories']['save_zip_directory']
    save_directory = config['directories']['save_directory']
    save_description = config['directories']['save_description']
    save_path = save_description + 'results.json'
    save_graph = config['directories']['save_graph']
    gexf_file = config['directories']['gexf_file']
    retrieval_nodes_path = config['directories']['retrieval_nodes_path']
    predef_graph = config['directories']['predefined_graph_path']

    isExist = os.path.exists(save_zip_directory)
    if not isExist:
        os.makedirs(save_zip_directory)
    isExist = os.path.exists(save_directory)
    if not isExist:
        os.makedirs(save_directory)
    isExist = os.path.exists(save_description)
    if not isExist:
        os.makedirs(save_description)


    random.seed(seed_no)


    # Load model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})


    signal.signal(signal.SIGINT, signal_handler)


    # Global States for User Preferences
    download_papers = gr.State(value=True)  # Default: Download papers
    train_model = gr.State(value=True)    # Default: Train the model


    # Categorized Recommended Prompts
    task_list = {
        "Abstract Completion",
        "Introduction to Abstract",
        "Title Generation",
        "Citation Recommendation",
        "Citation Sentence Generation",
        "Citation Link Prediction",
        "Influential Papers Recommendation",
        "Related Work Generation",
    }


    # CSS for Styling
    css = """
    body { background-color: #E0F7FA; margin: 0; padding: 0; }
    .gradio-container { background-color: #E0F7FA; border-radius: 10px; }
    #logo-container { display: flex; justify-content: center; align-items: center; margin: 0 auto; padding: 0; max-width: 120px; height: 120px; border-radius: 10px; overflow: hidden; }
    #scroll-menu { max-height: 310px; overflow-y: auto; padding: 10px; background-color: #fff; margin-top: 10px;}
    #task-header { background-color: #0288d1; color: white; font-size: 18px; padding: 8px; text-align: center; margin-bottom: 5px; margin-top: 40px; }
    #category-header { background-color: #ecb939; font-size: 16px; padding: 8px; margin: 10px 0; }
    """

    # State to store the selected task
    selected_task = gr.State(value="")


    # Gradio Interface
    with gr.Blocks(theme="soft", css=css) as demo:
        gr.HTML('<div id="logo-container"><img src="https://static.thenounproject.com/png/6480915-200.png" alt="Logo"></div>')
        gr.Markdown("# LitBench Interface")


        # Setup row for user preferences
        with gr.Row(visible=True) as setup_row:
            with gr.Column():
                gr.Markdown("### Setup Your Preferences")
                download_option = gr.Dropdown(
                    choices=["Download Paper", "Don't Download"],
                    value="Download Paper",
                    label="Download Option"
                )
                train_option = gr.Dropdown(
                    choices=["Train", "Don't Train"],
                    value="Train",
                    label="Training Option"
                )
                setup_button = gr.Button("Set Preferences and Proceed")


        # Chatbot row for user interaction
        with gr.Row(visible=False) as chatbot_row:
            # Store the currently selected task
            with gr.Column(scale=3):
                gr.Markdown("### Start Chatting!")
                chatbot = gr.ChatInterface(
                    predict,
                    chatbot=gr.Chatbot(height=400, type="messages", avatar_images=[
                        "https://icons.veryicon.com/png/o/miscellaneous/user-avatar/user-avatar-male-5.png",
                        "https://cdn-icons-png.flaticon.com/512/8649/8649595.png"
                    ]),
                    textbox=gr.Textbox(placeholder="Type your message here..."),
                    additional_inputs=selected_task,
                    additional_inputs_accordion=gr.Accordion(visible=False, label="Additional Inputs", ),
                )

                # Store user preferences and selected task for display
                preferences_output = gr.Textbox(value="", interactive=False, label="Your Preferences")

            # Task selection buttons for user interaction
            with gr.Column(scale=1):
                gr.HTML('<div id="task-header">Tasks:</div>')
                with gr.Column(elem_id="scroll-menu"):
                    # Create buttons
                    button_map = {prompt: gr.Button(prompt) for prompt in task_list}

                    for prompt in task_list:
                        button_map[prompt].click(
                            toggle_selection,
                            inputs=[selected_task, gr.State(value=prompt)],  # Toggle task selection
                            outputs=selected_task
                        ).then(
                            update_button_styles,  # Update button appearances
                            inputs=[selected_task],
                            outputs=[button_map[p] for p in task_list]  # Update all buttons
                        )


        # Setup button to finalize user preferences and start chatbot
        setup_button.click(
            setup,
            inputs=[download_option, train_option],
            outputs=[setup_row, chatbot_row, preferences_output, chatbot.chatbot]
        )


    # Launch the interface
    demo.launch(server_port=7880)
