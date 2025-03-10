from goodfire import Client, Variant

# Initialize Goodfire client
client = Client("sk-goodfire-5V7Yyoi0-yjkFdGtTxRbpNvru_TYjZHHCIu0KP09wiMprPydYXtEgg")
variant = Variant("meta-llama/Llama-3.3-70B-Instruct")


# Define prompts
test_prompt = [{"role": "user", "content": "What is the capital of France?"}]
true_statement = [{"role": "user", "content": "The capital of France is Paris."}]
false_statement = [{"role": "user", "content": "The capital of France is Berlin."}]


# Search for knowledge-related features (focus on factual knowledge)
knowledge_features = []
for query in ["capital cities", "countries and capitals", "geography", "facts about France", "political geography"]:
    knowledge_features.extend(client.features.search(query, model=variant, top_k=10))  


# Inspect activations for True and False
inspector_true = client.features.inspect(true_statement, model=variant, features=knowledge_features)
inspector_false = client.features.inspect(false_statement, model=variant, features=knowledge_features)

# Step 3: Compute activation differences
feature_differences = {}
for f_true, f_false in zip(inspector_true.top(k=50), inspector_false.top(k=50)):  # Examine top 50 for broad scope
    diff = abs(f_true.activation - f_false.activation)
    feature_differences[f_true.feature] = diff


# Select top N most sensitive features (e.g., top 8)
top_n = 8
sorted_features = sorted(feature_differences.items(), key=lambda x: x[1], reverse=True)
top_sensitive_features = [feat for feat, _ in sorted_features[:top_n]]

print("\nTop sensitive factual features selected for tracing:")
for feat, diff in sorted_features[:top_n]:
    print(f"- {feat.label}: diff {diff:.3f}")


# Create interventions (multi-feature)
# Erased suppress all top features
erased_variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
for feature in top_sensitive_features:
    erased_variant.set(feature, -0.5)  # Gentle suppression

# Amplified: enhance all top features
amplified_variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
for feature in top_sensitive_features:
    amplified_variant.set(feature, 0.3)  # Gentle boost


# Generate outputs
response_original = client.chat.completions.create(messages=test_prompt, model=variant)
response_erased = client.chat.completions.create(messages=test_prompt, model=erased_variant)
response_amplified = client.chat.completions.create(messages=test_prompt, model=amplified_variant)


# Step 7: Print results for comparison
print("\nOriginal Response:\n", response_original.choices[0].message["content"].strip())
print("\nAll Top Features Erased Response:\n", response_erased.choices[0].message["content"].strip())
print("\nAll Top Features Amplified Response:\n", response_amplified.choices[0].message["content"].strip())


