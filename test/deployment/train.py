import scalarlm

scalarlm.api_url = "https://qwen32b.i-blaze.com"

def get_dataset():
    dataset = []

    count = 1

    for i in range(count):
        dataset.append({"input": f"What is {i} + {i}?", "output": str(i + i)})

    return dataset


llm = scalarlm.SupermassiveIntelligence()

dataset = get_dataset()

status = llm.train(
    dataset,
    train_args={"max_steps": 3, "learning_rate": 3e-3, "gpus": 1, "nodes" : 2,
	    "gradient_accumulation_steps": 1,
            "max_token_block_size": 4096,
            "steps_per_checkpoint": 10000},
)

print(status)
