# **Title Generation Task - Input Format**

## **Overview**
The **Title Generation** task predicts the **title** of a research paper based on its **abstract**. The user provides the **abstract**, and the model generates a suitable title.

## **Expected Input Format**
The user input must strictly follow the format below:

```
Abstract: <abstract>
```

- `<abstract>`: The full abstract of the paper.

### **Example Input**
```
Abstract: This paper investigates the frequency behavior in spontaneous connected speech of two optional syntactic processes, particle movement and complementizer deletion. It shows them to be sensitive both to internal linguistic factors and to perceived norms of the standard language. It further compares the pattern found in usage with answers to a brief prescriptive grammatical questionnaire, where it finds parallelism. There is also a result of interest to the general theory of quantitative variation in an interaction found between an internal semantic effect and the external sociolinguistic one.
```

This formatted prompt is then used as input for the model to generate the predicted title of the paper.