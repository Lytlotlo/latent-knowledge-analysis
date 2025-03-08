import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval()  # Set model to evaluation mode

# Function to get activations for a given prompt
def get_activations(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # List of hidden states at each layer
    return hidden_states

# Define two test prompts: a factual statement and a possible misinformation
true_prompt = "The capital of France is Paris."
false_prompt = "The capital of France is Berlin."

# Get activations for both prompts
true_activations = get_activations(true_prompt)
false_activations = get_activations(false_prompt)

# Compute activation magnitudes for each layer
true_layer_activations = [torch.norm(layer, dim=-1).mean().item() for layer in true_activations]
false_layer_activations = [torch.norm(layer, dim=-1).mean().item() for layer in false_activations]

# Plot activation magnitudes per layer for both statements
plt.figure(figsize=(10, 5))
plt.plot(range(len(true_layer_activations)), true_layer_activations, marker="o", linestyle="-", label="True Statement")
plt.plot(range(len(false_layer_activations)), false_layer_activations, marker="x", linestyle="--", label="False Statement")
plt.xlabel("Layer Number")
plt.ylabel("Activation Magnitude")
plt.title("Activation Magnitudes Across GPT-2 Layers")
plt.legend()
plt.grid()
plt.show()

# Identify layers with the highest difference in activation between truth and falsehood
activation_differences = np.abs(np.array(true_layer_activations) - np.array(false_layer_activations))
most_sensitive_layer = np.argmax(activation_differences)
print(f"Layer with highest activation difference: {most_sensitive_layer}")
