from datasets import load_dataset
dataset = load_dataset("json", data_files="C:/Users/devas/MyDesktop/VS-Code/Legal/ipc_data.json")


def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

dataset = dataset.map(lambda x: {"text": format_example(x)})


#---------------
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = r"C:\Users\devas\MyDesktop\Tinyllma"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust based on model architecture
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Tokenize the dataset
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset["train"].map(tokenize, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
training_args = TrainingArguments(
    output_dir="./tinyllama-ipc-finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    save_total_limit=2,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator
)

trainer.train()


trainer.save_model("./tinyllama-ipc-finetuned")
tokenizer.save_pretrained("./tinyllama-ipc-finetuned")


model = AutoModelForCausalLM.from_pretrained("./tinyllama-ipc-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./tinyllama-ipc-finetuned")
