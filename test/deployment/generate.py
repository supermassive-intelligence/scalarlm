import masint


# masint.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-fastapi-app.modal.run"
# masint.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"


def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append(f"What is {i} + {i}? ")

    return dataset


llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

results = llm.generate(
    prompts=dataset,
    #model_name = "masint/tiny-random-llama"
    #model_name="e423033bf2955b65bfcfa77cc7b2e4892e7bdd14cdca27ed79c1f2da4edace03",
)

print(results)
