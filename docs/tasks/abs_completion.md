# **Abstract Completion Task - Input Format**

## **Overview**
The **Abstract Completion** task completes the abstract of a research paper. The user provides the **title** and a **partial abstract**, and the model completes the remaining abstract based on the given information.

## **Expected Input Format**
The user input must strictly follow the format below:

```
Title: <title>

Abstract: <abstract>
```

- `<title>`: The title of the paper.
- `<abstract>`: The partial abstract provided for completion.

### **Example Input**
```
Title: Attention Is All You Need

Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration.
```

This formatted prompt is then used as input for the model to generate the completed abstract.