import pandas as pd
from datasets import Dataset,DatasetDict
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from transformers import TrainingArguments
import os
df = pd.read_parquet("conversations.parquet")
trainDf = df.sample(frac = 0.9)
testDf = df.drop(trainDf.index)
trainDf.reset_index(drop = True,inplace = True)
testDf.reset_index(drop = True,inplace = True)
import re

TotalConversations = []
mapping = {
    'Doctor':'assistent',
    'Patient':'user'
}

pattern = "(Doctor|Patient):\s*(.*?)(?=\n+(?:Doctor|Patient):|\Z)"
for idx in range(len(trainDf)):
    convo = trainDf["conversation"][idx]
#     print(convo)
    ConvoList = []
    matches = re.findall(pattern,convo)
    for items in matches:
        ConvoList.append({
            'content':items[1],
            'role':mapping[items[0]]
        })
    if len(ConvoList) > 0:
        TotalConversations.append(ConvoList)

TotalConversationsTest = []
for idx in range(len(testDf)):
    convo = testDf["conversation"][idx]
#     print(convo)
    ConvoList = []
    matches = re.findall(pattern,convo)
    for items in matches:
        ConvoList.append({
            'content':items[1],
            'role':mapping[items[0]]
        })
    if len(ConvoList) > 0:
        TotalConversationsTest.append(ConvoList)

TrainConversations = pd.DataFrame()
TrainConversations["conversations"] = TotalConversations
# TrainConversations["conversations"][:5]
TestConversations = pd.DataFrame()
TestConversations["cpnversations"] = TotalConversationsTest

datasetTrain = Dataset.from_pandas(TrainConversations)
datasetTest = Dataset.from_pandas(TestConversations)
# remove this when done debugging
#indices = range(0,len(data))




dataset_dict = {"train": datasetTrain,
               "test": datasetTest}

raw_datasets = DatasetDict(dataset_dict)
print("Train dataset size:", len(raw_datasets["train"]))
print("Test dataset size:", len(raw_datasets["test"]))
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct" # use the model of your requirement

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Set reasonable default for models without max length
if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 4096


# Set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistent' %}\n{{ '<|assistent|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistent|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

import random
from multiprocessing import cpu_count
import torch

def apply_chat_template(example, tokenizer):
    messages = example["conversations"]
    #print(len(messages))
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

column_names = list(raw_datasets["train"].features)
column_names
# Check if a GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use the GPU if available, otherwise use the CPU
raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=torch.cuda.device_count() if torch.cuda.is_available() else cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template"
                                )

# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

#for index in random.sample(range(len(raw_datasets["train"])), 3):
  #print(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")


quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)
from trl import SFTTrainer
from peft import LoraConfig

# Assuming the model and tokenizer are already initialized earlier in your code
# model = ...  # Your model initialization
# tokenizer = ...  # Your tokenizer initialization

# Path where the Trainer will save its checkpoints and logs
os.makedirs(f'data/{model_id}-sft-lora', exist_ok=True)
output_dir = f'data/{model_id}-sft-lora'
# Initialize training arguments
training_args = TrainingArguments(
    fp16=False,  # specify bf16=True when training on GPUs that support bf16
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=128,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2.0e-05,
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    max_steps=-1,
    num_train_epochs=1,
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    save_strategy="epoch",
    save_total_limit=None,
    seed=42,
)

# LoRA configuration for fine-tuning
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Assuming you already have the train and eval datasets
# train_dataset = ...  # Your training dataset
# eval_dataset = ...  # Your evaluation dataset

# Initialize trainer
trainer = SFTTrainer(
    model=model_id,  # Reuse your already initialized model
    model_init_kwargs={},  # No need to pass model initialization kwargs here
    args=training_args,
    train_dataset=train_dataset,  # Ensure train_dataset is initialized
    eval_dataset=eval_dataset,  # Ensure eval_dataset is initialized
    dataset_text_field="text",
    tokenizer=tokenizer,  # Reuse your already initialized tokenizer
    packing=True,
    peft_config=peft_config,
    max_seq_length=tokenizer.model_max_length,
)

# Start training
train_result=trainer.train()

print("Training Completed")
metrics = train_result.metrics
#max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
#metrics["train_samples"] = min(max_train_samples, len(train_dataset))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
