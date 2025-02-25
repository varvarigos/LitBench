# **Bulk Evaluation for Literature-Based Tasks**

## **Overview**
`bulk_eval.py` is a script designed to **evaluate the performance of LLMs** on various literature-related tasks, including **citation sentence generation, link prediction, abstract completion, title generation, paper retrieval, and introduction-to-abstract generation**. The script provides a **batch evaluation pipeline** to assess models trained with **LitBench** datasets or other domain-specific literature datasets.

It loads a **citation graph dataset** and constructs evaluation prompts for the defined tasks. The script then uses the specified LLM to generate predictions and compares them against ground-truth outputs using **BERTScore and accuracy metrics**.

---

## **Usage**
To run `bulk_eval.py`, execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python3.10 src/bulk/bulk_eval.py \
    -config_path=conf/config_bulk_eval.yaml \
    -model=lora \
    -lorapath=models/llama_1b_qlora_uncensored_1_adapter_test_graph
```

## **Command-Line Arguments**

- `config_path`: Path to the configuration file for bulk evaluation.
- `model`: Model type (e.g., lora).
- `lorapath`: Path to the LLM model checkpoint.
- `index`: Index of the checkpoint to use for evaluation.

---

## **Supported Evaluation Tasks**
The script evaluates model performance across six key literature-based tasks:
1. Citation Sentence Generation (test_sentence)
* Generates a citation sentence describing how Paper A cites Paper B in the related work section.
* Evaluates output coherence using BERTScore.
2. Citation Link Prediction (test_LP)
* Determines if Paper A is likely to cite Paper B based on their titles and abstracts.
* Evaluates binary classification accuracy.
3. Abstract Completion (test_abs_completion)
* Completes a partially given abstract using the model’s understanding.
* Evaluates precision, recall, and F1-score using BERTScore.
4. Title Generation (test_title_generate)
* Predicts a paper’s title based on its abstract.
* Evaluates BERTScore similarity with ground-truth titles.
5. Citation Recommendation (test_retrival_e)
* Given a paper and a set of candidate papers, selects the one most likely to be cited.
* Evaluates retrieval accuracy.
6. Introduction to Abstract (test_intro_2_abs)
* Predicts a paper’s abstract based on its introduction section.
* Evaluates BERTScore similarity.

---

## Dependencies

Ensure you have the required Python libraries installed, following the instructions in [README.md](../../README.md)
