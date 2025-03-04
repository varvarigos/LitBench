# **Related Work Generation Task - Input Format**

## **Overview**
The Related Work Generation task generates a structured related work section based on the title and abstract of a given paper. The model analyzes the provided content and produces a coherent paragraph discussing relevant literature and prior work.


## **Expected Input Format**
The user input must strictly follow the format below:

```
Title of Paper: <title of the paper>

Abstract of Paper: <abstract of the paper>
```

- `<title of the paper>`: The title of the paper for which related work is being generated.

- `<abstract of the paper>`: The abstract of the paper, summarizing its key contributions and findings.


### **Example Input**
```
Title of Paper: Criticality in Tissue Homeostasis: Models and Experiments

Abstract of Paper: There is considerable theoretical and experimental support to the proposal that tissue homeostasis in the adult skin can be represented as a critical branching process. The homeostatic condition requires that the proliferation rate of the progenitor (P) cells (capable of cell division) is counterbalanced by the loss rate due to the differentiation of a P cell into differentiated (D) cells so that the total number of P cells remains constant. We consider the two-branch and three-branch models of tissue homeostasis to establish homeostasis as a critical phenomenon. It is first shown that some critical branching process theorems correctly predict experimental observations. A number of temporal signatures of the approach to criticality are investigated based on simulation and analytical results. The analogy between a critical branching process and mean-field percolation and sandpile models is invoked to show that the size and lifetime distributions of the populations of P cells have power-law forms. The associated critical exponents have the same magnitudes as in the cases of the mean-field lattice statistical models. The results indicate that tissue homeostasis provides experimental opportunities for testing critical phenomena.
```

This formatted prompt is then used as input for the model to generate a related work section discussing relevant literature and prior research related to the given paper.
