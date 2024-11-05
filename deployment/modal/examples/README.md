# Modal

1. You should have received an invite to modal. Accept the invite to create an account.

2. Login into modal.com

3. Go to the workspace apps page (https://modal.com/apps/smi-workspace/main) and click on "Quickstart Guide" near the search bar and follow the instructions to setup the modal CLI on your laptop.

4. Read through the short tutorials: 

https://modal.com/playground/get_started
https://modal.com/playground/custom_container
https://modal.com/playground/scaling_out


## Example: Text-to-sql Fine-tuning 

Install requirements:
```
cd smi-platform/deployment/modal/examples/text-to-sql-finetuning
pip install -r requirements.txt
```

Run training:
```
export ALLOW_WANDB=true 
modal run --detach src.train --config=config/mistral-memorize.yml --data=data/sqlqa.subsample.jsonl
```

Run inference on fine-tuned model:
```
modal run -q src.inference --prompt "[INST] Using the schema context below, generate a SQL query that answers the question.
CREATE TABLE head (name VARCHAR, born_state VARCHAR, age VARCHAR)
List the name, born state and age of the heads of departments ordered by name. [/INST]"
```

The streaming output from training should give a link to a WANDB dashboard for the run.

