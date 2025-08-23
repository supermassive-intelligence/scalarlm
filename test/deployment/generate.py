import scalarlm

scalarlm.api_url = "http://localhost:8000"

def get_dataset(count):
    dataset = []

    for i in range(count):
        dataset.append(f"What is {i} + {i}?")

    return dataset


llm = scalarlm.SupermassiveIntelligence()

dataset = ["Hello, how are you?", "What is 10 + 15?", "Tell me a story"]

results = llm.generate(
    prompts=dataset,
    max_tokens=200,
    model_name="584f8bc8bfaf6ee2d9a5a0616ee8994122ec1320a9059f4e4c563172be64c915",
)

print(results)
