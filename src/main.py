import pandas as pd
import numpy as np
import os
#import keras
import re
import tensorflow
import transformers
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from datasets import dataset_dict, load_metric,load_dataset
from transformers import pipeline
from transformers import Text2TextGenerationPipeline
from transformers import DataCollatorForSeq2Seq
import string
from transformers import Seq2SeqTrainer
from transformers import TFAutoModelForSeq2SeqLM
import tensorflow.python.ops.numpy_ops.np_config as np_config
import torch

#np_config.enable_numpy_behavior()
metric = load_metric("sacrebleu")

max_input_length =256
max_target_length =256
tokenizer=1
dataset = load_dataset('csv', data_files=os.path.join("data/en-fr.csv"))
model=1
splitted_dataset=dataset_dict

device = "cuda:1" if torch.cuda.is_available() else "cpu"
def preprocess_function(examples):
    inputs =examples["en"]
    targets = examples["fr"]
    global tokenizer
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    #print(labels)
    model_inputs["labels"] = labels["input_ids"]

    print("Exited preprocessing")
    return model_inputs

def load_data()->pd.DataFrame:
    file_path=os.path.join("data/en-fr.csv")
    df=pd.read_csv(file_path,nrows=20000)
    #print(df.head(1))
    #print(type(dataset))
   # print(dataset.shape)
    #print(len(dataset))
    #print(dataset.column_names)
    #print(dataset)
    global splitted_dataset
    splitted_dataset=dataset["train"].select(range(6000)).train_test_split(train_size=0.7,seed=2022)
    splitted_dataset["validation"] = splitted_dataset.pop("test")


        # splitted_dataset["train"][i]["fr"]=splitted_dataset["train"][i].apply(lambda x: str(x).lower())
        # splitted_dataset["train"][i]["fr"]=splitted_dataset["train"][i].apply(lambda x: re.sub("'","",x))
        # splitted_dataset["train"][i]["fr"]=splitted_dataset["train"][i].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        # splitted_dataset["train"][i]["fr"]=splitted_dataset["train"][i].apply(lambda x: x.translate(digit))
    splitted_dataset=splitted_dataset.filter(lambda x:x["en"] is not None)
    splitted_dataset=splitted_dataset.filter(lambda x:x["fr"] is not None)
   # print(splitted_dataset)
    #print(splitted_dataset["train"][1]["en"])
    return df

def load_pretrained_model():
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr",return_tensors="tf")
    model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
    #model =AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

    translator = pipeline("translation", model=model,tokenizer=tokenizer)
    #print(translator("Default to expanded threads"))
    #print(translator("Hello man"))


# def prepare_dataset(df:pd.DataFrame):
#     english_part:pd.Series = df['en']
#     french_part:pd.Series = df['fr']
#     print(english_part[1])

#     model_inputs = tokenizer(english_part[1], max_length=max_input_length, truncation=True)

def test_pipeline():
    translator = pipeline("translation", model=modeltf)
    print(translator("Default to expanded threads"))

def data_collation(tokenized_datasets):
    global model
    global tokenizer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
   # print(batch.keys())
   # print(batch["labels"])
   # print("-------------------------------------")
    #print(batch["decoder_input_ids"])
    return data_collator

def apply_preprocessing():
    global splitted_dataset
    tokenized_datasets = splitted_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=splitted_dataset["train"].column_names,)
    return tokenized_datasets


def create_tf_dataset(tokenized_dataset,data_collator):
    tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
    )
    tf_eval_dataset = tokenized_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=16,
    )

def compute_metrics(eval_preds):
    global tokenizer
    global model
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}



def train():
    global model
    global tokenized_datasets
    global tokenizer
    global data_collator
    args = Seq2SeqTrainingArguments(
        f"models/custom-en-to-fr",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        #fp16=True,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    )
    print(f"BEFORE TRAINING {trainer.evaluate(max_length=max_target_length)}")
    trainer.train()
    print(f"AFTER TRAINING{trainer.evaluate(max_length=max_target_length)}")

def predict():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path=AutoModelForSeq2SeqLM.from_pretrained(os.path.join("models/custom-en-to-fr/checkpoint-22"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("models/custom-en-to-fr/checkpoint-22"),return_tensors="tf")
    pipe=Text2TextGenerationPipeline(model=model_path,tokenizer=tokenizer)
    print(pipe("I love this movie!"))

if __name__ == '__main__':

    df=load_data().dropna(inplace=True)
    load_pretrained_model()
    tokenized_datasets= apply_preprocessing()
    data_collator=data_collation(tokenized_datasets)
    create_tf_dataset(tokenized_datasets,data_collator)
    train()

    #predict()
    #print(compute_metrics(tokenized_datasets,data_collator,eval))
    #prepare_dataset(df)
   # test_pipeline()
