import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
model.eval()  # Set model to evaluation mode

# Function to get attention weights for a given layer
def get_attention_weights(prompt, layer_num):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention weights for the chosen layer
    attentions = outputs.attentions[layer_num].squeeze(0).detach().clone()
    return attentions  # Shape: (num_heads, seq_len, seq_len)

# Function to generate text after modifying attention heads
def generate_text_from_modified_attention(original_prompt, layer, modification_type):
    inputs = tokenizer(original_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_attentions=True)

    # Replace attention weights for the selected layer
    modified_attentions = list(outputs.attentions)

    if modification_type == "zero":
        modified_attentions[layer] = torch.zeros_like(modified_attentions[layer])  # Disable attention
    elif modification_type == "boost":
        modified_attentions[layer] *= 10  # Strengthen attention

    # Pass modified attention through the model again
    with torch.no_grad():
        new_outputs = model(inputs["input_ids"], output_attentions=True)

    generated_text = tokenizer.decode(new_outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)
    return generated_text

# Define test prompt
original_prompt = "The capital of France is Paris."

# Get the number of layers
num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else 12  # Default to 12 if not found
layers_to_test = list(range(max(0, num_layers - 4), num_layers))  # Test last 4 layers

print(f"🔍 Testing layers: {layers_to_test}")

for layer in layers_to_test:
    print(f"\n🔎 **Testing Layer {layer} by Modifying Attention Heads**")
    
    # Generate responses using full-attention interventions
    zeroed_response = generate_text_from_modified_attention(original_prompt, layer, "zero")
    boosted_response = generate_text_from_modified_attention(original_prompt, layer, "boost")

    # Print Results
    print("Zeroed Attention Response (Disabled Attention): ", zeroed_response)
    print("Boosted Attention Response (Strengthened Attention): ", boosted_response)
