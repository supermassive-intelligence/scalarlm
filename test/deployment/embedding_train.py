import scalarlm

scalarlm.api_url = "http://localhost:8000"


def get_dataset():
    dataset = []

    count = 100

    for i in range(count):
        dataset.append(
            {"sentence1": f"What is {i} + {i}?", "sentence2": [f"{i + i}"], "score": 1.0}
        )
        dataset.append(
            {"sentence1": f"What is {i} + {i}?", "sentence2": [f"{i * i}"], "score": 0.0}
        )

    return dataset


llm = scalarlm.SupermassiveIntelligence()

dataset = get_dataset()

status = llm.train(
    dataset,
    train_args={
        "max_steps": 10,
        "learning_rate": 3e-3,
        "gpus": 1,
        "max_token_block_size": 32,
        "steps_per_checkpoint": 10000,
        "batch_size": 16,
    },
)

print(status)
