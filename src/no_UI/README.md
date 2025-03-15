# **1. Fine-Tuning for Literature-Based LLMs (without User Interface)**

## **Overview**
[`finetune_noUI.py`](finetune_noUI.py) is a script designed to **fine-tune large language models (LLMs) on literature-based tasks** using **QLoRA**. The script supports **training domain-specific models** for citation reasoning, abstract generation, retrieval, and more. It leverages **LoRA adapters** to enable efficient fine-tuning on consumer-grade GPUs.

The script reads **a citation graph dataset**, constructs training prompts for multiple tasks, and fine-tunes an LLM using the QLoRA framework. The resulting **LoRA-adapted model** can then be used for inference or further training.

---

## **Usage**
To run `finetune_noUI.py`, execute the following command:

```bash
python3.10 src/no_UI/finetune_noUI.py configs/config_noUI.yaml --index 1
```

## **Command-Line Arguments**
- config_path: Path to the YAML configuration  
- index: Index specifying GPU/task number (default: 1).

## **Supported Fine-Tuning Tasks**
The script fine-tunes LLMs on seven key literature-based tasks, generating instruction-tuned training data:

1. **Citation Sentence Generation:** Trains the model to generate citation sentences describing how Paper A cites Paper B in the related work section.

2. **Citation Link Prediction:** Trains the model to predict whether Paper A is likely to cite Paper B based on their titles and abstracts.

3. **Abstract Completion:** Trains the model to complete an abstract given a partial abstract and a paper title.

4. **Title Generation:** Trains the model to generate a paper’s title based on its abstract.

5. **Citation Recommendation:** Trains the model to select the most relevant paper from a set of candidates that Paper A is likely to cite.

6. **Introduction to Abstract:** Trains the model to generate an abstract based on a paper’s introduction.

---

## Dependencies

Ensure you have the required Python libraries installed, following the instructions in [README.md](../../README.md)

---

# **2. Evaluation for Literature-Based Tasks (Without User Interface)**

## **Overview**
[`eval_noUI.py`](eval_noUI.py) is a script designed to **evaluate the performance of LLMs** on various literature-related tasks, including **citation sentence generation, link prediction, abstract completion, title generation, paper retrieval, and introduction-to-abstract generation**. The script provides a **batch evaluation pipeline** to assess models trained with **LitBench** datasets or other domain-specific literature datasets.

It loads a **citation graph dataset** and constructs evaluation prompts for the defined tasks. The script then uses the specified LLM to generate predictions and compares them against ground-truth outputs using **BERTScore and accuracy metrics**.

---

## **Usage**
To run `eval_noUI.py`, execute the following command:

```bash
python3.10 src/no_UI/eval_noUI.py \
    -config_path=configs/config_noUI.yaml \
    -adapter_path=models/llama_1b_qlora_uncensored_1_adapter_test_graph
```

## **Command-Line Arguments**

- `config_path`: Path to the configuration file for evaluation.
- `adapter_path`: Path to the LLM model checkpoint.

---

## **Supported Evaluation Tasks**
The script evaluates model performance across six key literature-based tasks:
1. Citation Sentence Generation (`test_sentence`)
    * Generates a citation sentence describing how Paper A cites Paper B in the related work section.
    * Evaluates output coherence using BERTScore.

2. Citation Link Prediction (`test_LP`)
    * Determines if Paper A is likely to cite Paper B based on their titles and abstracts.
    * Evaluates binary classification accuracy.

3. Abstract Completion (`test_abs_completion`)
    * Completes a partially given abstract using the model’s understanding.
    * Evaluates precision, recall, and F1-score using BERTScore.

4. Title Generation (`test_title_generate`)
    * Predicts a paper’s title based on its abstract.
    * Evaluates BERTScore similarity with ground-truth titles.

5. Citation Recommendation (`test_retrival`)
    * Given a paper and a set of candidate papers, selects the one most likely to be cited.
    * Evaluates retrieval accuracy.

6. Introduction to Abstract (`test_intro_2_abs`)
    * Predicts a paper’s abstract based on its introduction section.
    * Evaluates BERTScore similarity.

---

## Dependencies

Ensure you have the required Python libraries installed, following the instructions in [README.md](../../README.md)
