from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
import os
from openai import OpenAI
import anthropic
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BitsAndBytesConfig
from transformers import pipeline, set_seed
import time
from random import choice
from tqdm import tqdm
from accelerate import Accelerator

def generate_text_gpt4(prompts, num_completions=5, temperature=1):
    
    client = OpenAI()  # Checking for Access Key 

    results = []
    for prompt in tqdm(prompts):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "الرجاء إكمال الجملة التالية . أجب بكلمة واحدة أو إكمال قصير للجملة. لا تفسيرات. الجملة:"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=len(prompt)+ 10,
            n=num_completions  # Generate multiple completions
        )
        # wait 1 second
        
        
        # Collect all completions for the prompt
        completions = [choice.message.content for choice in completion.choices]
        results.append(completions)
        
    return results


# API call to Claude-3.5-sonnet model
def generate_text_claude(prompt):
    client = anthropic.Anthropic(
        api_key="my_api_key",
    )
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content

# load the model gemini-1.5-flash
def load_mode_gemini():
    genai.configure(api_key='')
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    return model

# loading a local model from huggingface
def load_model_hugg(model_id):
    print(f"Loading model {model_id}")
    torch.cuda.set_device(0)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Enable 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 for faster computations
        bnb_4bit_use_double_quant=True,  # Optional: Improves accuracy slightly
        bnb_4bit_quant_type="nf4"        # Optional: Use NormalFloat4 quantization
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if model_id == "CohereForAI/aya-101" or model_id == "google/flan-t5-base":
     

        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, low_cpu_mem_usage=True,quantization_config=quantization_config,
                                                     trust_remote_code=True, device_map={'':torch.cuda.current_device()})
        accelerator = Accelerator(device_placement=True, cpu=False)
        print(torch.cuda.current_device())
        model, tokenizer = accelerator.prepare(model, tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True,
                                                     quantization_config=quantization_config,
                                                     trust_remote_code=True)
        
    

    model.eval()
    return model, tokenizer

# generate text from gemini model
def generate_text_gemini(model, prompt):
   
    response = model.generate_content(prompt)
    return response.text

def generate_text_t5(model, tokenizer, prompts, sys_prompt, k=5, max_new_tokens=10, t=0.7):
    

    # Encode the input prompt for the T5 model
    prompts =list(prompts)
    prompts = [sys_prompt + prompt for prompt in prompts]

    input_ids = tokenizer(prompts, return_tensors="pt",padding=True,
        truncation=True,
        max_length=512)["input_ids"].to(model.device)

    
    
    
    # Generate output using the encoder-decoder structure
    outputs = model.generate(input_ids=input_ids, max_length=input_ids.size(1) + max_new_tokens, num_return_sequences=k, temperature=t, do_sample=True,)
    
    outputs_grouped = outputs.view(input_ids.size(0), k, -1)  # Reshape to (batch_size, num_return_sequences, sequence_length)

    # Decode each group of sequences
    generated_texts_grouped = []
    for group in outputs_grouped:
        generated_texts_grouped.append(tokenizer.batch_decode(group, skip_special_tokens=True))
    
    return generated_texts_grouped

# generate text from huggingface model
def generate_text_hugg(model, tokenizer, prompts, k=5, max_new_tokens=15, t=0.7):
    """
    Generate text using Hugging Face's pipeline for text generation.

    Args:
        name_model: The name or path of the model.
        prompts: List of input prompts.
        k: Number of sequences to generate per prompt.
        max_new_tokens: Number of additional tokens to generate beyond the input length.
        t: Temperature for sampling.

    Returns:
        List of generated texts.
    """
    device = torch.device("cuda:1")
    #model = model.to(device)

    # Initialize the pipeline
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    # Set seed for reproducibility
    set_seed(42)

    # Generate text for each prompt
    filled_templates = [
        [fill['generated_text'] for fill in generator(
            prompt,
            max_length=len(tokenizer.encode(prompt)) + max_new_tokens,
            num_return_sequences=k,
            do_sample=True
        )]
        for prompt in prompts
    ]

    return filled_templates

