from goodfire import Client, Variant

# Initialize Goodfire client
client = Client("sk-goodfire-5V7Yyoi0-yjkFdGtTxRbpNvru_TYjZHHCIu0KP09wiMprPydYXtEgg")
variant = Variant("meta-llama/Llama-3.3-70B-Instruct")


# Define prompts
true_statement = [{"role": "user", "content": "The capital of France is Paris."}]
false_statement = [{"role": "user", "content": "The capital of France is Berlin."}]


#  Search for factual knowledge features
knowledge_features = []
for query in ["capital cities", "countries and capitals", "geography", "facts about France", "political geography"]:
    knowledge_features.extend(client.features.search(query, model=variant, top_k=10))

#
# Inspect activations for True and False
inspector_true = client.features.inspect(true_statement, model=variant, features=knowledge_features)
inspector_false = client.features.inspect(false_statement, model=variant, features=knowledge_features)

#Compute activation differences
feature_differences = {}
for f_true, f_false in zip(inspector_true.top(k=50), inspector_false.top(k=50)):
    diff = abs(f_true.activation - f_false.activation)
    feature_differences[f_true.feature] = diff


# Select top N most sensitive features
top_n = 5  # Focused set for deep tracing
sorted_features = sorted(feature_differences.items(), key=lambda x: x[1], reverse=True)
top_sensitive_features = [feat for feat, _ in sorted_features[:top_n]]

print("\n Top sensitive factual features selected for deep tracing:")
for feat, diff in sorted_features[:top_n]:
    print(f"- {feat.label}: diff {diff:.3f}")


# Trace dependencies for each feature
for idx, feature in enumerate(top_sensitive_features, start=1):
    print(f"\n🔍 Tracing for Feature #{idx}: {feature.label}")

    # Find upstream features (what activates this feature)
    upstream = client.features.neighbors(feature, model=variant, top_k=10)
    print("  ↑ Upstream features (activating this feature):")
    for f in upstream:
        print(f"    - {f.label}")

    # Find downstream features (what this feature activates)
    downstream = client.features.neighbors(feature, model=variant, top_k=10)  
    print("  ↓ Downstream features (activated by this feature):")
    for f in downstream:
        print(f"    - {f.label}")



