import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval()  # Set model to evaluation mode

# Function to get activations at a specific layer
def get_layer_activations(prompt, layer_num):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # List of hidden states at each layer
    return hidden_states[layer_num].squeeze(0).detach().clone()  # Clone to avoid modifying original

# Function to generate text given activations
def generate_text_from_activations(activations, original_prompt):
    inputs = tokenizer(original_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    # Replace original activations with modified ones
    modified_hidden_states = list(outputs.hidden_states)
    modified_hidden_states[most_sensitive_layer] = activations  # Apply modified activations

    # Pass modified activations through the model again
    with torch.no_grad():
        new_outputs = model(inputs["input_ids"], output_hidden_states=True)

    generated_text = tokenizer.decode(new_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return generated_text

# Define test prompt
original_prompt = "The capital of France is Paris."

# Identify the most sensitive layer and neuron
most_sensitive_layer = 10  # Based on previous analysis
most_sensitive_neuron = 5   # The neuron with highest activation difference

# Get original activations
original_activations = get_layer_activations(original_prompt, most_sensitive_layer)

# Create two modified versions:
# 1. Set Neuron 5’s activation to zero (erasing knowledge)
zeroed_activations = original_activations.clone()
zeroed_activations[:, most_sensitive_neuron] = 0  # Zeroing out Neuron 5

# 2. Amplify Neuron 5’s activation (enhancing knowledge)
amplified_activations = original_activations.clone()
amplified_activations[:, most_sensitive_neuron] *= 2  # Amplify Neuron 5

# Generate text using modified activations
original_response = generate_text_from_activations(original_activations, original_prompt)
zeroed_response = generate_text_from_activations(zeroed_activations, original_prompt)
amplified_response = generate_text_from_activations(amplified_activations, original_prompt)

# Print Results
print("Original Response: ", original_response)
print("Zeroed Response (Neuron 5 = 0): ", zeroed_response)
print("Amplified Response (Neuron 5 * 2): ", amplified_response)
