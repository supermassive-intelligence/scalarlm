import masint
import asyncio

masint.api_url = "https://greg1232--cray-cpu-llama-3-2-1b-fastapi-app.modal.run"
# masint.api_url = "https://greg1232--cray-nvidia-llama-3-2-3b-instruct-fastapi-app.modal.run"

count = 10
def get_dataset():
    dataset = []

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}", "output": "The answer is " + str(i + i)}
        )

    return dataset

llm = masint.SupermassiveIntelligence()

dataset = get_dataset()

status = asyncio.run(llm.train(dataset, train_args={"max_steps": (100*count)}))

print(status)

