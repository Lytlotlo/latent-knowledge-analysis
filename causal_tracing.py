from goodfire import Client, Variant

client = Client("sk-goodfire-5V7Yyoi0-yjkFdGtTxRbpNvru_TYjZHHCIu0KP09wiMprPydYXtEgg")
variant = Variant("meta-llama/Llama-3.3-70B-Instruct")

# Define prompts
test_prompt = [{"role": "user", "content": "What is the capital of France?"}]
true_statement = [{"role": "user", "content": "The capital of France is Paris."}]
false_statement = [{"role": "user", "content": "The capital of France is Berlin."}]

# Search for factual knowledge features
knowledge_features = []
for query in ["capital cities", "countries and capitals", "geography", "facts about France", "political geography"]:
    knowledge_features.extend(client.features.search(query, model=variant, top_k=10))


# Inspect activations for True statement
inspector_true = client.features.inspect(true_statement, model=variant, features=knowledge_features)

# Filter for active features
active_features = [f for f in inspector_true.top(k=20) if f.activation > 0.1]

# Inspect activations for False statement
inspector_false = client.features.inspect(false_statement, model=variant, features=[f.feature for f in active_features])

#Compute differences
feature_differences = {}
for f_true, f_false in zip(inspector_true.top(k=20), inspector_false.top(k=20)):
    diff = abs(f_true.activation - f_false.activation)
    feature_differences[f_true.feature] = diff

# Select most sensitive feature
most_sensitive_feature = max(feature_differences, key=feature_differences.get)
print(f"🔥 Most sensitive factual feature: {most_sensitive_feature.label} with diff {feature_differences[most_sensitive_feature]:.3f}")


#Intervene carefully (gentle adjustments)
erased_variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
erased_variant.set(most_sensitive_feature, -0.5)  # Gentle erase

amplified_variant = Variant("meta-llama/Llama-3.3-70B-Instruct")
amplified_variant.set(most_sensitive_feature, 0.3)  # Gentle amplify


# Get outputs
response_original = client.chat.completions.create(messages=test_prompt, model=variant)
response_erased = client.chat.completions.create(messages=test_prompt, model=erased_variant)
response_amplified = client.chat.completions.create(messages=test_prompt, model=amplified_variant)


# Print and compare
print("\nOriginal Response:\n", response_original.choices[0].message["content"].strip())
print("\nFeature Erased Response:\n", response_erased.choices[0].message["content"].strip())
print("\nFeature Amplified Response:\n", response_amplified.choices[0].message["content"].strip())
