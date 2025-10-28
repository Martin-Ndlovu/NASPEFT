from transformers import Trainer, AutoTokenizer, LlamaAdapterModel
from datasets import load_from_disk
tokenizer = AutoTokenizer.from_pretrained("../models/Meta-Llama-3.1-8B-Instruct")
model = LlamaAdapterModel.from_pretrained("output/wikitext")
dataset = load_from_disk("../datasets/wikitext/wikitext-103-v1")
eval_dataset = dataset["validation"].select(range(100))
def preprocess_function(examples):
    result = tokenizer(examples["text"], padding="max_length", max_length=256, truncation=True)
    result["labels"] = result["input_ids"].copy()
    return result
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
trainer = Trainer(model=model, eval_dataset=eval_dataset)
metrics = trainer.evaluate()
print(metrics)