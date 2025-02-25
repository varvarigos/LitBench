# **Citation Link Prediction Task - Input Format**

## **Overview**
The **Citation Link Prediction** task determines whether **Paper A** will cite **Paper B**. The user provides the **titles and abstracts** of both papers, and the model predicts **"Yes"** or **"No"** based on their relevance.

## **Expected Input Format**
The user input must strictly follow the format below:

```
Title A: <title of Paper A>

Abstract A: <abstract of Paper A>

Title B: <title of Paper B>

Abstract B: <abstract of Paper B>
```

- `<title of Paper A>`: The title of Paper A (the potential citing paper).
- `<abstract of Paper A>`: The abstract of Paper A.
- `<title of Paper B>`: The title of Paper B (the potential cited paper).
- `<abstract of Paper B>`: The abstract of Paper B.

### **Example Input**
```
Title A: A Study on Machine Learning

Abstract A: This paper presents a study on machine learning algorithms.

Title B: A Survey on Deep Learning

Abstract B: This paper presents a survey on deep learning algorithms.
```

After receiving this input, the model will analyze the relationship between **Paper A and Paper B** and provide a **direct answer**:

- **"Yes"** → If Paper A is likely to cite Paper B.
- **"No"** → If Paper A is unlikely to cite Paper B.

This formatted prompt is then used as input for the model to determine the citation relationship.
