from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
raw_datasets=load_dataset('glue','sst2')

#glue benchmark for SENTIMENT ANALYSIS

#print(raw_datasets)
 #prints 
'''
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 67349
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 872
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1821
    })
})'''

#print(dir(raw_datasets['train']))
 #prints all functions and attributes e.g. train_test, sort,to_list,to_pandas etc

print(raw_datasets['train'][:2]) #prints 2 sentences, 2 labels, 2 indexes

print(raw_datasets['train'].features)

checkpoint = 'distilbert-base-uncased' #faster

tokenizer= AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences = tokenizer(raw_datasets['train'][:3]['sentence'])

from pprint import pprint
pprint(tokenized_sentences)


def tokenize_fn(batch):
    return tokenizer(batch['sentence'],truncation=True)

tokenized_datasets=raw_datasets.map(tokenize_fn,batched=True)

from transformers import TrainingArguments

training_args = TrainingArguments('my_trainer',evaluation_strategy='epoch'
                                  ,save_strategy='epoch',
                                  num_train_epochs=1)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)

from torchinfo import summary

print(summary(model))

from transformers import Trainer

params_before = []
for name,p in model.named_parameters():
    params_before.append(p.detach().cpu().numpy())


from datasets import load_metric

metric=load_metric('glue','sst2')

print(metric.compute(predictions=[1,0,1],references=[1,0,0]))

trainer=Trainer(model,training_args,train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=tokenizer)

trainer.train()
trainer.save_model('saved_model_5may')

from transformers import pipeline

newmodel=pipeline('text-classification',model='saved_model_5may',device=0)

print(newmodel('this movie is bad'))