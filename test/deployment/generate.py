import scalarlm

scalarlm.api_url = "http://localhost:8000"


llm = scalarlm.SupermassiveIntelligence()

dataset = ["Hello, how are you?", "What is 10 + 15?", "Tell me a story"]

results = llm.generate(
    prompts=dataset,
    max_tokens=32,
)

print(results)
