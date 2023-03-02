from djl_python import Input, Output
import deepspeed
import torch
import logging
import math
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer



def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location, load_in_8bit=True)
   
    model = AutoModelForCausalLM.from_pretrained(
        model_location, 
        device_map="balanced_low_0", 
        load_in_8bit=True
    )
    model.requires_grad_(False)
    model.eval()
    
    return model, tokenizer


model = None
tokenizer = None
generator = None


def run_inference(model, tokenizer, data, params):
    generate_kwargs = params
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(data,
                                               return_tensors="pt",
                                               padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    outputs = model.generate(**input_tokens, **generate_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    input_sentences = data["inputs"]
    params = data["parameters"]
    
    outputs = run_inference(model, tokenizer, input_sentences, params)
    result = {"outputs": outputs}
    return Output().add_as_json(result)
