# SparrowVQE: Visual Question Explanation for Course Content Understanding

A Custom 3B parameter Model Enhanced for Educational Contexts: This specialized model integrates slide-text pairs from machine learning classes, leveraging a unique training approach. It connects a frozen pre-trained vision encoder (SigLip) with a frozen language model (Phi-2) through an innovative projector. The model employs attention mechanisms and language modeling loss to deeply understand and generate educational content, specifically tailored to the context of machine learning education.

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650c7fbb8ffe1f53bdbe1aec/DTjDSq2yG-5Cqnk6giPFq.jpeg" width="40%" height="auto"/>
</p>


<div class="center-div", align="center">
    <table width="100%" height="auto">
        <tr>
            <td align="center">
                <a href="https://colab.research.google.com/github/rrymn/SparrowVQE/blob/main/SparrowVQE_Demo.ipynb">[Google Colab Demo]</a>
                <a href="https://huggingface.co/spaces/rrymn/SparrowVQE">[🤗 HuggingFace Demo]</a>
            </td>
        </tr>
    </table>
</div>


## Table of Contents

- [Introduction](#introduction)
- [Project Description](#project-description)
- [Visual Question Answering (VQA) and Visual Question Explanation (VQE)](#visual-question-answering-vqa-and-visual-question-explanation-vqe)
  - [Visual Question Answering (VQA)](#visual-question-answering-vqa)
  - [Visual Question Explanation (VQE)](#visual-question-explanation-vqe)
- [Training Process](#training-process)
- [Applications & Benefits](#applications--benefits)
- [How to Use](#how-to-use)
- [Results](#results)



## Introduction
In the rapidly evolving field of educational technology, students often encounter difficulties in grasping complex concepts, particularly in subjects like machine learning. These challenges are exacerbated when students rely on static study materials, such as slides and textbooks, which lack interactive or explanatory feedback. Addressing this gap, our research introduces the Machine Learning Visual Question Explanation (MLVQE) dataset. This innovative dataset represents a significant step forward in the realm of Visual Question Answering (VQA), specifically designed to aid students in better understanding their course materials through advanced AI-driven explanations.

## Project Description
The MLVQE dataset is meticulously compiled from a machine learning course and comprises 885 slide images coupled with 110,407 words from lecture transcripts, structured into 9,416 question-answer pairs. At the heart of our project is the SparrowVQE model, a pioneering VQA system that synthesizes the capabilities of two advanced models, SigLIP and Phi-2. This model undergoes a rigorous three-stage training regimen: multimodal pre-training to understand and integrate different forms of data, instruction tuning to align with specific educational goals, and domain fine-tuning to tailor its responses to the field of machine learning. This comprehensive training strategy allows SparrowVQE to effectively merge and interpret both visual and textual inputs, significantly boosting its ability to provide detailed, context-aware explanations. With its superior performance on the MLVQE dataset and its success in surpassing existing VQA benchmarks, SparrowVQE substantially enhances students' interaction with and comprehension of visual course material, offering a more engaging and informative learning experience.

## Visual Question Answering (VQA) VS Visual Question Explanation (VQE)
### Visual Question Answering (VQA)
- **Objective**: The primary goal of VQA systems is to provide concise, accurate answers to questions based on visual data such as images or videos. The answers are usually short and direct, focusing on identifying and stating facts visible in the visual input.

- **Functionality**: VQA models process an input image and a related question, using techniques from both computer vision and natural language processing to interpret the content of the image and the intent of the question before providing an appropriate answer.

- **Output**: Outputs are generally brief and limited to the information explicitly requested in the question. For example, if asked "What color is the car?" the VQA system would simply respond with the color.

- **Applications**: VQA systems are commonly used in environments where quick factual answers are required, such as identifying objects in images for accessibility purposes or automated customer service tools.

### Visual Question Explanation (VQE)
- **Objective**: The aim of VQE systems, like the SparrowVQE model, is not only to answer questions based on visual data but to also provides personlizated explanations like a professor that will enhance understanding.

- **Functionality**: VQE models extend the capabilities of VQA by incorporating more sophisticated AI techniques that enable them to analyze both the visual elements and the textual data associated with them more comprehensively. This includes integrating data from related texts, such as lecture transcripts or explanatory notes, to enrich the quality of the response.

- **Output**: The output from a VQE system is more expansive and informative. It provides not just the answer but also an explanation that helps the user understand the reasoning behind the answer or additional context about the subject matter, in our case majorly focussing on the personalisation.

- **Applications**: VQE systems are particularly useful in educational settings, where understanding and context are crucial. They help students grasp complex subjects by not only answering specific questions about visual materials (like slides from a lecture) but also explaining concepts and processes in detail.

While VQA focuses on answering questions directly from visual inputs, VQE aims to provide personalised responses like professor that help deepen the user's understanding of both the question and the visual content, making it particularly valuable in educational contexts where detailed explanations are essential for learning.


## Training Process
The model underwent a three-stage distributed training regimen, primarily conducted on eight A100 80GB GPUs. Utilizing SLURM as the job scheduling system, the training sessions typically lasted between 9 to 11 hours for each stage.

## Applications & Benifits
Our initial focus has been on developing and refining the MLVQE (Machine Learning Visual Question Explanation) dataset through a meticulous three-stage training process. While our current dataset is specifically tailored to a machine learning class taught by one of our professors—including his transcripts and question-answer pairs it sets the groundwork for broader applications. We envision expanding this model to accommodate various subjects, creating specialized datasets like DLVQE (Deep Learning Visual Question Explanation), NLPVQE (Natural Language Processing Visual Question Explanation), and even extending into fields such as cybersecurity and biotechnology.

### The potential applications of our model include:

- **Personalization of Online Learning**: Our approach personalizes online courses by providing specific, tailored responses to learners’ questions about course material. This closes the gap between students and professors, making remote learning more interactive and personal.

- **Enhanced Student Engagement**: By offering instant, customized responses, our model helps students grasp complex topics more effectively, thereby improving engagement and comprehension.

- **Scalability Across Disciplines**: The versatility of our model allows it to be adapted for multiple courses, offering a robust tool for educational institutions to enhance their online learning platforms.

These features collectively work towards transforming traditional online education into a more dynamic and responsive experience for students across various disciplines.



## How to use

**Install dependencies**
```bash
pip install transformers 
pip install -q pillow accelerate einops
```


```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

torch.set_default_device("cuda")

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "rrymn/SparrowVQE", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("rrymn/SparrowVQE", trust_remote_code=True)

#function to generate the answer
def predict(question, image_path):
    #Set inputs
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question}? ASSISTANT:"
    image = Image.open(image_path)
    
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to('cuda')
    image_tensor = model.image_preprocess(image)
    
    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=25,
        images=image_tensor,
        use_cache=True)[0]
    
    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

```
# Results

## Week 3-8 Slides Summary
![Week 3 to Week 8 Machine Learning Concepts](/images/example_01.drawio.png)
*This image provides a visual summary of key machine learning concepts covered between weeks 3 to 8. It includes topics such as avoiding overfitting, understanding logistic functions in the context of probabilities, exploring the 'face space' in image recognition, analyzing the curse of dimensionality, PCA as matrix factorization, and Gaussian Mixture Models.*

## Week 10-15 Slides Summary
![Week 10 to Week 15 Machine Learning Concepts](/images/example_02.drawio.png)
*This image provides a visual summary of key machine learning concepts covered between weeks 10 to 15. It illustrates topics such as the differences in model spaces between decision trees and nearest neighbors, understanding margins in SVMs, the role of Vision Transformers and MLP heads in neural networks, the effect of bagging on model variance, and an introduction to entropy in the context of decision trees.*
