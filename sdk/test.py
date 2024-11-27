import masint
import asyncio

count = 10
def get_dataset():
    dataset = []

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}", "output": "The answer is " + str(i + i)}
        )

    return dataset

llm = masint.AsyncSupermassiveIntelligence()

dataset = get_dataset()

status = asyncio.run(llm.train(dataset, train_args={"max_steps": (100*count)}))

print(status)

