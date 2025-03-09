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

# Function to generate text given modified activations
def generate_text_from_activations(activations, original_prompt, layer):
    inputs = tokenizer(original_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    # Replace original activations with modified ones
    modified_hidden_states = list(outputs.hidden_states)
    modified_hidden_states[layer] = activations  # Apply modified activations

    # Pass modified activations through the model again
    with torch.no_grad():
        new_outputs = model(inputs["input_ids"], output_hidden_states=True)

    generated_text = tokenizer.decode(new_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return generated_text

# Define test prompt
original_prompt = "The capital of France is Paris."

# Test multiple layers (expanding beyond Layer 10)
layers_to_test = [9, 10, 11, 12]  # Different layers may store knowledge

for layer in layers_to_test:
    print(f"\n🔎 **Testing Layer {layer}**")
    
    # Get activations for the chosen layer
    original_activations = get_layer_activations(original_prompt, layer)
    
    # Compute absolute differences for each neuron
    neuron_differences = np.abs(original_activations.mean(axis=0).flatten())  # Ensure it's 1D
    
    # Get top 5 neurons with the highest activation differences
    num_neurons = neuron_differences.shape[0]
    top_neurons = np.argsort(neuron_differences)[-min(5, num_neurons):]

    # Create two modified versions:
    zeroed_activations = original_activations.clone()
    amplified_activations = original_activations.clone()

    for neuron in top_neurons:
        zeroed_activations[:, neuron] = 0  # Zeroing out top neurons
        amplified_activations[:, neuron] *= 2  # Amplifying top neurons

    # Generate text using modified activations
    original_response = generate_text_from_activations(original_activations, original_prompt, layer)
    zeroed_response = generate_text_from_activations(zeroed_activations, original_prompt, layer)
    amplified_response = generate_text_from_activations(amplified_activations, original_prompt, layer)

    # Print Results
    print("🔹 Original Response: ", original_response)
    print("❌ Zeroed Response (Top Neurons = 0): ", zeroed_response)
    print("✅ Amplified Response (Top Neurons * 2): ", amplified_response)
