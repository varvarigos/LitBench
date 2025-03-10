# **Citation Recommendation Task - Input Format**

## **Overview**
The **Citation Recommendation** task selects the **most likely paper to be cited** by **Paper A** from a list of **candidate papers**. The user provides the **title and abstract of Paper A**, along with the **titles of candidate papers**, and the model predicts which paper is most likely to be cited.

## **Expected Input Format**
The user input must strictly follow the format below:

```
Title of the Paper A: <title>
Abstract of the Paper A: <abstract>
candidate papers:
0. <title>
1. <title>
2. <title>
...
```

- `<title>`: The title of Paper A.
- `<abstract>`: The abstract of Paper A.
- **Candidate papers** are listed as `<candidate number>. <title>`.

### **Example Input**
```
Title of the Paper A: A New Method for Finding Shortest Paths in Graphs
Abstract of the Paper A: This paper presents a new method for solving the problem of finding the shortest path between two points in a graph. The method is based on a new algorithm that is more efficient than existing algorithms. The paper also presents experimental results that show the effectiveness of the new method.
candidate papers:
0. A Survey of Shortest Path Algorithms
1. An Improved Algorithm for Finding Shortest Paths
2. A Comparison of Shortest Path Algorithms
```

This formatted prompt is then used as input for the model to select the **most relevant paper** that Paper A is likely to cite.
