### CV split - Binary Classification - mhc motifs
# 

import os
import numpy as np
import pandas as pd 
from pathlib import Path
from scipy.special import softmax
from tqdm import tqdm

from datasets import load_from_disk,Dataset,DatasetDict
from transformers import (
    DebertaForSequenceClassification,
    DebertaTokenizerFast,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from aim.hugging_face import AimCallback

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    pred_scores = softmax(logits, axis=1)[:, 1]
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions, zero_division=0)
    recall = recall_score(y_true=labels, y_pred=predictions, zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    auc = roc_auc_score(y_true=labels, y_score=pred_scores, average='micro')
    aupr = average_precision_score(y_true=labels, y_score=pred_scores, average='micro')

    return {
        "accuracy": accuracy,
        "auc": auc,
        "aupr": aupr,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

access_token="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"
max_length=288

model_name='tcrhlamotifs-crossencoder'  
tokenizer_name= model_name.split("-")[0]
ds_name='fs22_cdrcv_mismhc'

ds_path=f'/data/finetuning/01-BinaryClassification/fullseq/cv/tokenized_datasets/{ds_name}_{tokenizer_name}'

cv_folds=os.listdir(ds_path)
cv_binary_results=[]
for fold in tqdm(cv_folds):
    
    # Initialize aim_callback
    aim_callback = AimCallback(experiment=f'{model_name}_{ds_name}_{fold}', repo='/data/aim')

    model = DebertaForSequenceClassification.from_pretrained(f'shepherdgroup/{model_name}',num_labels=2,token=access_token)

    tokenizer = DebertaTokenizerFast.from_pretrained(f'/data/finetuning/tokenizers/{tokenizer_name}',max_len=max_length)
    new_tokens=["[cdra25]","[cdrb25]"]
    tokenizer.add_tokens(list(new_tokens))
    model.resize_token_embeddings(len(tokenizer)) # for added cdr25 token
    tokenized_datasets = load_from_disk(f'{ds_path}/{fold}')
    tokenized_datasets.set_format('torch')

    training_args = TrainingArguments(
        output_dir=f"models/{ds_name}/{model_name}/{fold}",
        logging_dir=f"logs/{ds_name}/{model_name}/{fold}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        evaluation_strategy='steps',
        eval_steps=100,
        logging_steps=100,
        num_train_epochs=4, 
        weight_decay=0.2, #0.2
        warmup_ratio=0.15,
        lr_scheduler_type='cosine',
        learning_rate=2e-5, #2e-5
        save_strategy='steps',
        save_total_limit=1,
        push_to_hub=False,
        report_to='all',
        seed=42,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['eval'],
        callbacks=[aim_callback],
        compute_metrics=compute_metrics
    )

    print(f"Training a {model_name} model on {fold}")
    trainer.train()
    predictions = trainer.predict(tokenized_datasets['test'])
    
    result_metric=predictions.metrics
    result_metric['fold']=fold
    cv_binary_results.append(result_metric)
    
    predictions = softmax(predictions.predictions, axis=1)[:, 1]
    
    pred_df = pd.DataFrame(predictions, columns=['prediction'])
    pred_df['label'] = tokenized_datasets['test']['label']
    pred_df['predicted_label']=(pred_df['prediction'] >0.5)/1.
    outpath = f"results/{ds_name}/{model_name}"
    Path(outpath).mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(f'{outpath}/{fold}.csv', index=False)
    
final_binary_results=pd.DataFrame(cv_binary_results)
print(final_binary_results)
pred_path=f"/data/finetuning/01-BinaryClassification/fullseq/cv/predictions"
final_binary_results.to_csv(f'{pred_path}/{ds_name}_{model_name}.csv',index=False)
print(f"completed {ds_name}")