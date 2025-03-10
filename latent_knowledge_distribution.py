import numpy as np
import plotly.graph_objects as go
from goodfire import Client, Variant
import random

# Initialize Goodfire client
client = Client("sk-goodfire-5V7Yyoi0-yjkFdGtTxRbpNvru_TYjZHHCIu0KP09wiMprPydYXtEgg")
variant = Variant("meta-llama/Llama-3.3-70B-Instruct")


# Define feature concepts relevant to factual knowledge
feature_terms = [
    "knowledge of capitals",
    "geography",
    "political facts",
    "world knowledge",
    "reasoning about countries",
    "city facts",
    "country leaders",
    "government facts"
]

# Define the fact to analyze (True and False versions)
true_prompt = [{"role": "user", "content": "The capital of France is Paris."}]
false_prompt = [{"role": "user", "content": "The capital of France is Berlin."}]
rag_prompt = [{"role": "user", "content": "The capital of France is Berlin. (Please check a knowledge base.)"}]
target_answer = "Paris"


# Data storage for plotting
x_labels, x_labels_idx, y_activations, z_differences, sizes, colors = [], [], [], [], [], []

# Process each feature concept
for i, feature_term in enumerate(feature_terms):
    features = client.features.search(feature_term, model=variant, top_k=1)
    if not features:
        print(f"Feature not found: {feature_term}")
        continue

    # Inspect activations for True, False, and RAG cases
    inspector_true = client.features.inspect(true_prompt, model=variant, features=features)
    inspector_false = client.features.inspect(false_prompt, model=variant, features=features)
    inspector_rag = client.features.inspect(rag_prompt, model=variant, features=features)

    # Get top feature activation values
    true_act = inspector_true.top(k=1)[0].activation
    false_act = inspector_false.top(k=1)[0].activation
    rag_act = inspector_rag.top(k=1)[0].activation

    # Activation difference (truth sensitivity) and RAG dependency
    activation_diff = abs(true_act - false_act)
    rag_dependency = abs(rag_act - false_act)

    # Model's confidence in correct answer (probability for "Paris")
    logits = client.chat.logits(
        messages=true_prompt,
        model=variant,
        filter_vocabulary=[target_answer]
    )
    prob = logits.logits.get(target_answer, 0.0)

    # Add jitter for horizontal spacing
    jitter = random.uniform(-0.2, 0.2)
    x_labels_idx.append(i + jitter)  

    # Rescale bubble size 
    bubble_size = 5 + 15 * prob  

    # Save for plotting
    x_labels.append(feature_term)
    y_activations.append(true_act)
    z_differences.append(activation_diff)
    sizes.append(bubble_size)
    colors.append(rag_dependency)  

# Prepare 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x_labels_idx,  
    y=y_activations,  
    z=z_differences,  
    mode='markers',
    marker=dict(
        size=sizes, 
        color=colors,  
        colorscale='RdYlBu_r', 
        opacity=0.7, 
        line=dict(width=1, color='DarkSlateGrey'),  
        colorbar=dict(title='RAG Dependency (Activation Change)')  # Color legend
    ),
    text=[
        f"Feature: {label}<br>"
        f"Activation (True): {act:.3f}<br>"
        f"Activation Diff (T/F): {diff:.3f}<br>"
        f"Confidence (Paris): {((size - 5)/15):.3f}<br>"
        f"RAG Dependency: {color:.3f}"
        for label, act, diff, size, color in zip(x_labels, y_activations, z_differences, sizes, colors)
    ]  # Hover text for clarity
)])


# Final layout adjustments for readability
fig.update_layout(
    title='4D Visualization of Factual Knowledge Sensitivity and RAG Dependence (Refined)',
    scene=dict(
        xaxis_title='Feature (Concept)',
        yaxis_title='Activation on True Statement',
        zaxis_title='Activation Difference (Truth vs Falsehood)',
        xaxis=dict(tickvals=list(range(len(x_labels))), ticktext=x_labels),  
    ),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=50),
    font=dict(size=12)
)


# Show the interactive plot
fig.show()

