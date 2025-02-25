# **Citation Sentence Generation Task - Input Format**

## **Overview**
The **Citation Sentence** task generates a citation sentence describing how **Paper A** cites **Paper B**. The user provides the **titles and abstracts of both papers**, and the model generates a structured citation sentence.

## **Expected Input Format**
The user input must strictly follow the format below:

```
Title A: <title of Paper A>

Abstract A: <abstract of Paper A>

Title B: <title of Paper B>

Abstract B: <abstract of Paper B>
```

- `<title of Paper A>`: The title of Paper A (the citing paper).
- `<abstract of Paper A>`: The abstract of Paper A.
- `<title of Paper B>`: The title of Paper B (the cited paper).
- `<abstract of Paper B>`: The abstract of Paper B.

### **Example Input**
```
Title A: A Neural Algorithm of Artistic Style

Abstract A: We present an algorithm that generates artistic images of high perceptual quality. The algorithm uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images.

Title B: Image Style Transfer Using Convolutional Neural Networks

Abstract B: We describe a method for transferring the style of one image onto the content of another image using convolutional neural networks. The method is based on matching the feature representations of the content and style images in a high-level convolutional neural network.
```

This formatted prompt is then used as input for the model to generate a **citation sentence** describing how **Paper A references Paper B**.
