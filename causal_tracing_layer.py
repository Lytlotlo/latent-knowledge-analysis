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
    
    # Ensure the layer exists
    if layer_num >= len(outputs.hidden_states):
        raise ValueError(f"❌ Layer {layer_num} does not exist in this GPT-2 model (max layer = {len(outputs.hidden_states) - 1})")
    
    return outputs.hidden_states[layer_num].squeeze(0).detach().clone()  # Clone to avoid modifying original

# Function to generate text given modified activations
def generate_text_from_modified_layer(original_prompt, layer, modification_type):
    inputs = tokenizer(original_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)

    # Replace entire layer activations with modifications
    modified_hidden_states = list(outputs.hidden_states)

    if modification_type == "zero":
        modified_hidden_states[layer] = torch.zeros_like(modified_hidden_states[layer])  # Set layer to zero
    elif modification_type == "noise":
        modified_hidden_states[layer] = torch.randn_like(modified_hidden_states[layer]) * 10  # Inject random noise
    
    # Pass modified activations through the model again
    with torch.no_grad():
        new_outputs = model(inputs["input_ids"], output_hidden_states=True)

    generated_text = tokenizer.decode(new_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return generated_text

# Define test prompt
original_prompt = "The capital of France is Paris."

# Get the correct number of layers
num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else 12  # Default to 12 if not found
layers_to_test = list(range(max(0, num_layers - 4), num_layers))  # Test last 4 layers

print(f"🔍 Testing layers: {layers_to_test}")

for layer in layers_to_test:
    print(f"\n🔎 **Testing Layer {layer} by Overwriting Entire Activations**")
    
    # Generate responses using full-layer interventions
    zeroed_response = generate_text_from_modified_layer(original_prompt, layer, "zero")
    randomized_response = generate_text_from_modified_layer(original_prompt, layer, "noise")

    # Print Results
    print("❌ Zeroed Layer Response (Full Layer = 0): ", zeroed_response)
    print("🎲 Randomized Layer Response (Full Layer = Noise): ", randomized_response)
