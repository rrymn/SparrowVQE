# SparrowVQE


<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/650c7fbb8ffe1f53bdbe1aec/DTjDSq2yG-5Cqnk6giPFq.jpeg" width="40%" height="auto"/>
</p>

<a href="https://colab.research.google.com/github/rrymn/SparrowVQE/blob/main/SparrowVQE_Demo.ipynb" target="_blank">
<img src=https://colab.research.google.com/assets/colab-badge.svg style="margin-bottom: 5px;" />
</a>
<a href="https://huggingface.co/rrymn/SparrowVQE" target="_blank">
<img src=https://cdn-lfs.huggingface.co/repos/96/a2/96a2c8468c1546e660ac2609e49404b8588fcf5a748761fa72c154b2836b4c83/942cad1ccda905ac5a659dfd2d78b344fccfb84a8a3ac3721e08f488205638a0?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27hf-logo.svg%3B+filename%3D%22hf-logo.svg%22%3B&response-content-type=image%2Fsvg%2Bxml&Expires=1711041884&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMTA0MTg4NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy85Ni9hMi85NmEyYzg0NjhjMTU0NmU2NjBhYzI2MDllNDk0MDRiODU4OGZjZjVhNzQ4NzYxZmE3MmMxNTRiMjgzNmI0YzgzLzk0MmNhZDFjY2RhOTA1YWM1YTY1OWRmZDJkNzhiMzQ0ZmNjZmI4NGE4YTNhYzM3MjFlMDhmNDg4MjA1NjM4YTA%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=rejsI8okYYzYbpXSY541NRlomDv3Kmcvw8ssqgO-o45UEiCqgTdjSmzTRfJqx2EfMKBoEDCX3LqzK3tKP7cXbAcUP5RSwTR430teR8%7E63kOLAW5D1LYf6%7E31Vz7ArYi5WIO5D-BB1m6mXjahzbyXTlPESBRUJiik5paxSe3oGjN1E2Xk0j2YTt32aCWvZCqiVS7ztGe1uiG0kDu3oAiYXZcib%7EEFm8WfMaYXFa1kI5iSVfMQXl8MJw1wIdECU679rsWyzHl5D7PLKc17wJX9WlgDazBIceAXrVJDzOARIf1fKfyUsUioIa87oOnKB6GX1yN02OZkySmuYFt0chhG8w__&Key-Pair-Id=KVTP0A1DKRTAX style="margin-bottom: 5px;" />
</a>




<p align='center', style='font-size: 16px;' >A Custom 3B parameter Model Enhanced for Educational Contexts: This specialized model integrates slide-text pairs from machine learning classes, leveraging a unique training approach. It connects a frozen pre-trained vision encoder (SigLip) with a frozen language model (Phi-2) through an innovative projector. The model employs attention mechanisms and language modeling loss to deeply understand and generate educational content, specifically tailored to the context of machine learning education. </p>

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
## Week 3-8 Slides Summary
![Week 3 to Week 8 Machine Learning Concepts](/images/example_01.drawio.png)
*This image provides a visual summary of key machine learning concepts covered between weeks 3 to 8. It includes topics such as avoiding overfitting, understanding logistic functions in the context of probabilities, exploring the 'face space' in image recognition, analyzing the curse of dimensionality, PCA as matrix factorization, and Gaussian Mixture Models.*

## Week 10-15 Slides Summary
![Week 10 to Week 15 Machine Learning Concepts](/images/example_02.drawio.png)
*This image provides a visual summary of key machine learning concepts covered between weeks 10 to 15. It illustrates topics such as the differences in model spaces between decision trees and nearest neighbors, understanding margins in SVMs, the role of Vision Transformers and MLP heads in neural networks, the effect of bagging on model variance, and an introduction to entropy in the context of decision trees.*

