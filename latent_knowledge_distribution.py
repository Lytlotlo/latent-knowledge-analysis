import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import plotly.graph_objects as go

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can swap this for 'gpt2-medium' or others if needed
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
hidden_states_true = outputs_true.hidden_states  # (num_layers, batch, seq_len, hidden_size)
hidden_states_false = outputs_false.hidden_states

# Number of layers and neurons
num_layers = len(hidden_states_true)
hidden_size = hidden_states_true[0].shape[-1]  # Should be 768 for GPT-2 small

print(f"✅ Model has {num_layers} layers, each with {hidden_size} neurons")

# Average activations across sequence length for simplicity
def average_activations(hidden_states):
    return torch.stack([layer.mean(dim=1).squeeze(0) for layer in hidden_states])  # (num_layers, hidden_size)

avg_true = average_activations(hidden_states_true)  # (num_layers, hidden_size)
avg_false = average_activations(hidden_states_false)  # (num_layers, hidden_size)

# Compute absolute differences for each neuron in each layer
activation_differences = torch.abs(avg_true - avg_false).numpy()  # (num_layers, hidden_size)

# --- 3D Visualization Section ---

# Flatten data for structured 3D scatter plot
layer_indices = []
neuron_indices = []
differences = []

threshold = np.percentile(activation_differences, 95)  # Top 5% most active neurons

for layer in range(num_layers):
    for neuron in range(hidden_size):
        diff = activation_differences[layer][neuron]
        if diff >= threshold:  # Only include significant neurons
            layer_indices.append(layer)  # Y-axis: Layer
            neuron_indices.append(neuron)  # X-axis: Neuron
            differences.append(diff)  # Z-axis/Color/Size

# Normalize differences for size
max_diff = max(differences)
sizes = [15 + 30 * (d / max_diff) for d in differences]  # Ensure minimum size for visibility

# Structured 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=neuron_indices,
    y=layer_indices,
    z=differences,  # Could also do Z=layer and use color for diff if we want flat layers
    mode='markers',
    marker=dict(
        size=sizes,
        color=differences,  # Color by activation difference
        colorscale='Viridis',
        opacity=0.8,
        line=dict(width=0.5, color='DarkSlateGrey')  # Add outline for clarity
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
