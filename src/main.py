import pandas as pd
import os
import transformers
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from datasets import load_metric
from transformers import pipeline

metric = load_metric("sacrebleu")
modeltf=transformers.models.marian.modeling_marian.MarianMTModel
def tokenize_function(examples):
    pass

def load_data()->pd.DataFrame:
    file_path=os.path.join("data/en-fr.csv")
    df=pd.read_csv(file_path,nrows=10000)
    print(df.head(1))
    return df

def load_pretrained_model():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr",return_tensors="tf")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    modeltf=model
    translator = pipeline("translation", model=model,tokenizer=tokenizer)
    #print(translator("Default to expanded threads"))
    print(translator("Hello man"))
   


def test_pipeline():
    translator = pipeline("translation", model=modeltf)
    print(translator("Default to expanded threads"))

if __name__ == '__main__':
    df=load_data()
    load_pretrained_model()
   # test_pipeline()
