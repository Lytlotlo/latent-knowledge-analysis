import numpy as np
import matplotlib.pyplot as plt
from goodfire import Client, Variant

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
    "government facts",
    "historical facts",
    "culture"
]

# Define the fact to analyze (True and False versions)
true_prompt = [{"role": "user", "content": "The capital of France is Paris."}]
false_prompt = [{"role": "user", "content": "The capital of France is Berlin."}]


# Storage for activations
feature_labels = []
true_activations = []
false_activations = []

# Process each feature concept
for feature_term in feature_terms:
    features = client.features.search(feature_term, model=variant, top_k=1)
    if not features:
        print(f"Feature not found: {feature_term}")
        continue

    # Inspect activations for True and False statements
    inspector_true = client.features.inspect(true_prompt, model=variant, features=features)
    inspector_false = client.features.inspect(false_prompt, model=variant, features=features)

    # Extract top activation for that feature
    true_act = inspector_true.top(k=1)[0].activation
    false_act = inspector_false.top(k=1)[0].activation

    # Store data
    feature_labels.append(feature_term)
    true_activations.append(true_act)
    false_activations.append(false_act)


# Bar plot to compare activations
x = np.arange(len(feature_labels))  
width = 0.35  # Width of the bars

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, true_activations, width, label='True Fact (Paris)', color='green', edgecolor='black')
bars2 = ax.bar(x + width/2, false_activations, width, label='False Fact (Berlin)', color='red', edgecolor='black')

# Add labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Final formatting
ax.set_xlabel('Feature (Concept)')
ax.set_ylabel('Activation Value')
ax.set_title('Feature Activation Probing: True vs False Fact')
ax.set_xticks(x)
ax.set_xticklabels(feature_labels, rotation=30, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
