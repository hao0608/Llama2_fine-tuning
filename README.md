# Fine-Tuning LLaMA 2 with PEFT and QLoRA

## Project Overview

This project focuses on fine-tuning the large language model [LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), using the QLoRA (Quantized Low-Rank Adaptation) method within the Parameter-Efficient Fine-Tuning (PEFT) techniques. QLoRA significantly reduces the model's memory footprint while maintaining performance, enabling the fine-tuning and deployment of large models in resource-constrained environments.

## Project Workflow

### 1. Introduction to LLaMA 2

[LLaMA 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) is a large language model launched by Meta AI in February 2023. It has gained widespread popularity for being fully open-source and offering performance comparable to ChatGPT.

### 2. PEFT (Parameter-Efficient Fine-Tuning)

PEFT aims to minimize the computational resource requirements for fine-tuning large models, preserving the knowledge from the pre-training phase.

### 3. Fine-tuning Process

#### 3.1 Applying for LLaMA 2 Model Access

- Register on Hugging Face and apply for model access.
- Create a token to download model weights.

#### 3.2 Loading the Model with QLoRA

- Load model weights using 4-bit NormalFloat (nf4) quantization.
- Adjust the model using the `peft` package, applying QLoRA for optimization.

#### 3.3 Data Preparation

- Prepare the training dataset using PTT Chinese corpus, aiming to train a LLaMA 2 model adept at answering in Traditional Chinese.

#### 3.4 Model Training

- Split the dataset into training and validation sets, converting them to DataLoader format.
- Set up the optimizer and training plan, and observe the training results.

#### 3.5 RLHF (Reinforcement Learning from Human Feedback)

- Further refine model responses through Reinforcement Learning from Human Feedback.

## Technical Highlights

- **QLoRA**: A Quantized Low-Rank Adaptation technique that significantly reduces GPU memory consumption by quantizing the originally frozen parameters of the language model to 4-Bit.
- **PEFT**: Enhances parameter efficiency in fine-tuning, allowing fine-tuning with fewer parameters while retaining pre-trained knowledge.
- **RLHF**: Combines reinforcement learning and human feedback to optimize the model for more accurate and natural responses.

## Conclusion

This project demonstrates an efficient process for fine-tuning the LLaMA 2 model using QLoRA technology. By leveraging PEFT and QLoRA, we have successfully reduced the computational burden of training and deploying the model. Furthermore, we have enhanced the model's generative capabilities and stability through RLHF, providing a viable solution for fine-tuning large models in resource-limited settings.

