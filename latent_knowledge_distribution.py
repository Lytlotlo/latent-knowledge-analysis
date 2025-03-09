import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import plotly.graph_objects as go

# Load GPT-2 model and tokenizer
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
model.eval()  # Set model to evaluation mode

# Define true and false prompts
true_prompt = "The capital of France is Paris."
false_prompt = "The capital of France is Berlin."

# Tokenize inputs
inputs_true = tokenizer(true_prompt, return_tensors="pt")
inputs_false = tokenizer(false_prompt, return_tensors="pt")

# Run both prompts and extract hidden states (activations)
with torch.no_grad():
    outputs_true = model(**inputs_true, output_hidden_states=True)
    outputs_false = model(**inputs_false, output_hidden_states=True)

# Extract hidden states (tuple of tensors, one per layer)
hidden_states_true = outputs_true.hidden_states  
hidden_states_false = outputs_false.hidden_states

# Number of layers and neurons
num_layers = len(hidden_states_true)
hidden_size = hidden_states_true[0].shape[-1]  

print(f"Model has {num_layers} layers, each with {hidden_size} neurons")

# Average activations across sequence length for simplicity
def average_activations(hidden_states):
    return torch.stack([layer.mean(dim=1).squeeze(0) for layer in hidden_states])  

avg_true = average_activations(hidden_states_true)  
avg_false = average_activations(hidden_states_false)  

# Compute absolute differences for each neuron in each layer
activation_differences = torch.abs(avg_true - avg_false).numpy()  

# --- 3D Visualization Section ---

# Flatten data for structured 3D scatter plot
layer_indices = []
neuron_indices = []
differences = []

# Top 5% most active neurons
threshold = np.percentile(activation_differences, 95)  

for layer in range(num_layers):
    for neuron in range(hidden_size):
        diff = activation_differences[layer][neuron]
        if diff >= threshold:  
            layer_indices.append(layer)  
            neuron_indices.append(neuron)  
            differences.append(diff)  

# Normalize differences for size
max_diff = max(differences)
sizes = [15 + 30 * (d / max_diff) for d in differences]  

# Structured 3D scatter plot
# Could also do Z=layer and use color for diff if we want flat layers
fig = go.Figure(data=[go.Scatter3d(
    x=neuron_indices,
    y=layer_indices,
    z=differences,  
    mode='markers',
    marker=dict(
        size=sizes,
        color=differences,  
        colorscale='Viridis',
        opacity=0.8,
        line=dict(width=0.5, color='DarkSlateGrey')  
    )
)])

fig.update_layout(
    title='3D Visualization of Neurons Sensitive to Truth vs. Falsehood',
    scene=dict(
        xaxis=dict(title='Neuron Index (Per Layer)', backgroundcolor="rgb(200, 200, 230)", gridcolor="white"),
        yaxis=dict(title='Layer Index (0 = input, higher = deeper)', backgroundcolor="rgb(230, 200,230)", gridcolor="white"),
        zaxis=dict(title='Activation Difference', backgroundcolor="rgb(230, 230,200)", gridcolor="white"),
    ),
    width=1000,
    height=700,
    margin=dict(r=10, l=10, b=10, t=30)
)

fig.show()
