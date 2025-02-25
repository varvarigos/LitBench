# **Influential Papers Recommendation Task - Input Format**

## **Overview**
The **Influential Papers** task identifies the **K most influential papers** in a citation graph based on their number of citations. The user specifies the **number of papers (K)** to retrieve, and the model returns the **titles and abstracts** of the most cited papers.

## **Expected Input Format**
The user input must strictly follow the format below:

```
Number of papers to consider: <K>
```

- `<K>`: The number of most influential papers to retrieve.

### **Example Input**
```
Number of papers to consider: 3
```

This formatted prompt is then used as input for the model to generate a ranked list of the most influential papers.

### **Example Output**
```
Here are the most influential papers:
1. Title: Advances in Neural Networks
   Abstract: This paper presents a comprehensive survey of neural network architectures and training techniques.
   
2. Title: Deep Learning for Image Recognition
   Abstract: We introduce a novel deep learning model that achieves state-of-the-art accuracy on image classification tasks.
   
3. Title: Graph-Based Learning Methods
   Abstract: This work explores the use of graph-based algorithms for semi-supervised learning and knowledge extraction.
```
