from transformers import AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
import os
from openai import OpenAI
import anthropic
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BitsAndBytesConfig

# API call to GPT-4 model
def generate_text_gpt4(prompt):

    """
    INPUT: string of text that will be evaluated
    OUPUT: Content Output from the Model
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key = OPENAI_API_KEY) #checking for Access Key 

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ],
        logprobs = True
    )

    return completion.choices[0].message.content

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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,               # Enable 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 for faster computations
        bnb_4bit_use_double_quant=True,  # Optional: Improves accuracy slightly
        bnb_4bit_quant_type="nf4"        # Optional: Use NormalFloat4 quantization
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True,
                                                     quantization_config=quantization_config,
                                                     trust_remote_code=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)
    return model, tokenizer

# generate text from gemini model
def generate_text_gemini(model, prompt):
   
    response = model.generate_content(prompt)
    return response.text

# generate text from huggingface model
def generate_text_hugg(model, tokenizer, prompts, bs= 8, max_new_tokens=20):

    with torch.no_grad():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(prompts, return_tensors="pt", padding = True, truncation = True).to(device)
        outputs = model.module.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'],  
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

