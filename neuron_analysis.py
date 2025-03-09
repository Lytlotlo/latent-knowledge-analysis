import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

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
    return hidden_states[layer_num].squeeze(0).detach().numpy()  # Convert to numpy array

# Define two test prompts
true_prompt = "The capital of France is Paris."
false_prompt = "The capital of France is Berlin."

# Identify the most sensitive layer from previous results
most_sensitive_layer = 10  # Adjust based on previous graph

# Get neuron activations for the chosen layer
true_neurons = get_layer_activations(true_prompt, most_sensitive_layer)
false_neurons = get_layer_activations(false_prompt, most_sensitive_layer)

# Compute absolute differences for each neuron
neuron_differences = np.abs(true_neurons - false_neurons)

# If neuron_differences is multi-dimensional (e.g., shape (num_neurons, 768)), reduce to 1D
if len(neuron_differences.shape) > 1:
    neuron_differences = neuron_differences.mean(axis=1)  # Take the mean across dimensions

# Get the actual number of neurons in the selected layer
num_neurons = neuron_differences.shape[0]  

# Ensure we only select neurons that exist in range [0, num_neurons-1]
sorted_indices = np.argsort(neuron_differences)  # Sort neurons by activation difference
valid_indices = np.unique(sorted_indices[-min(10, num_neurons):])  # Select top neurons, ensuring uniqueness

# Debug: Print number of neurons available and selected indices
print(f"Total neurons in Layer {most_sensitive_layer}: {num_neurons}")
print(f"Selected neuron indices: {valid_indices}")

# Fixing out-of-bounds error by ensuring indices are within valid range
valid_neuron_differences = neuron_differences[valid_indices]

# Ensure valid_neuron_differences is 1D for plotting
valid_neuron_differences = np.array(valid_neuron_differences).flatten()

# Plot the differences for the top neurons
plt.figure(figsize=(10, 5))
plt.bar(range(len(valid_indices)), valid_neuron_differences, tick_label=[f"Neuron {i}" for i in valid_indices])
plt.xlabel("Neuron Index")
plt.ylabel("Activation Difference")
plt.title(f"Top {len(valid_indices)} Neurons with Highest Activation Differences (Layer {most_sensitive_layer})")
plt.xticks(rotation=45)
plt.grid()
plt.show()
