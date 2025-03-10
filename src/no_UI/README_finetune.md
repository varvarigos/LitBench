# **Fine-Tuning for Literature-Based LLMs (without User Interface)**

## **Overview**
`finetune_noUI.py` is a script designed to **fine-tune large language models (LLMs) on literature-based tasks** using **QLoRA**. The script supports **training domain-specific models** for citation reasoning, abstract generation, retrieval, and more. It leverages **LoRA adapters** to enable efficient fine-tuning on consumer-grade GPUs.

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
