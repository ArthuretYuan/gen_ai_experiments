# fine-tuned GPT-2 model optimized for question answering. (SQuAD)
# REF: https://github.com/omidiu/GPT-2-Fine-Tuning/blob/main/main.ipynb


#from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import math

# loading the SQuAD dataset
dataset = load_dataset("squad")
print(dataset['train'][0])



# loading the DistilGPT-2 tokenizer
model_checkpoint = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

special_tokens = tokenizer.special_tokens_map
print(special_tokens)


# preposessing the dataset
def add_end_token_to_question(input_dict):
    input_dict['question'] += special_tokens['bos_token']
    return input_dict
dataset = dataset.remove_columns(['id', 'title', 'context', 'answers'])
dataset = dataset.map(add_end_token_to_question)


# tokenizing the dataset using tokenizer
def tokenize_function(input_dict):
    return tokenizer(input_dict['question'], truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['question'])


# grouping tokenized text
max_block_length = 128

def divide_tokenized_text(tokenized_text_dict, block_size):
    """
    Divides the tokenized text in the examples into fixed-length blocks of size block_size.

    Parameters:
    -----------
    tokenized_text_dict: dict
        A dictionary containing tokenized text as values for different keys.

    block_size: int
        The desired length of each tokenized block.

    Returns:
    -----------
        dict: A dictionary with tokenized text divided into fixed-length blocks.
    """
    concatenated_examples = {k: sum(tokenized_text_dict[k], []) for k in tokenized_text_dict.keys()}
    total_length = len(concatenated_examples[list(tokenized_text_dict.keys())[0]])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result['labels'] = result['input_ids'].copy()
    return result


lm_dataset = tokenized_dataset.map(
    lambda tokenized_text_dict: divide_tokenized_text(tokenized_text_dict, max_block_length),
    batched=True,
    batch_size=1000,
    num_proc=4,
)


# get train and evaluation datasets and fine-tune the model
def finetune():
    train_dataset = lm_dataset['train'].shuffle(seed=42).select(range(100))
    eval_dataset = lm_dataset['validation'].shuffle(seed=42).select(range(100))
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    training_args = TrainingArguments(
        f'./{model_checkpoint}-squad',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False, # Change to True to push the model to the Hub
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print(f'Perplexity: {math.exp(eval_results["eval_loss"]):.2f}')
    tokenizer.save_pretrained('gpt2-squad')
