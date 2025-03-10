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
    "social studies",
    "culture"
]

# Define the fact to analyze (True and False versions)
true_prompt = [{"role": "user", "content": "The capital of France is Paris."}]
false_prompt = [{"role": "user", "content": "The capital of France is Berlin."}]

# Store results
feature_labels = []
activation_differences = []

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

    # Compute absolute difference as sensitivity
    diff = abs(true_act - false_act)

    # Store
    feature_labels.append(feature_term)
    activation_differences.append(diff)

# 
# Sort by activation difference (descending)
sorted_indices = np.argsort(activation_differences)[::-1]  
sorted_labels = [feature_labels[i] for i in sorted_indices]
sorted_diffs = [activation_differences[i] for i in sorted_indices]

# Limit to top N for clarity 
top_n = min(8, len(sorted_labels))  
final_labels = sorted_labels[:top_n]
final_diffs = sorted_diffs[:top_n]


# Plotting bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(final_labels, final_diffs, color='skyblue', edgecolor='black')

# Add activation difference as text above bars
for bar, diff in zip(bars, final_diffs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{diff:.3f}', ha='center', va='bottom', fontsize=10)

# Final formatting
plt.title('Top Features Sensitive to Factual Knowledge (Truth vs. Falsehood)')
plt.xlabel('Feature (Concept)')
plt.ylabel('Activation Difference')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

