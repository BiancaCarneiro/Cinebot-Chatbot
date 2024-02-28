import evaluate
import numpy as np
from pandas import DataFrame
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report
# https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt

MODEL = "bert-base-multilingual-uncased"

class TrainTransformers():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    def __init__(self, train_df:DataFrame, test_df:DataFrame, val_df:DataFrame, label_list:list) -> None:
        self.train_df = Dataset.from_pandas(train_df)
        self.test_df = Dataset.from_pandas(test_df)
        self.val_df = Dataset.from_pandas(val_df)
        self.label_list = label_list
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=128, padding=True)

        labels = []
        for i, label in enumerate(examples["label"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def train(self, model_name:str, lr:float=0.001, num_epochs:int=10, batch_size:int=500):
        train_data = self.train_df.map(self.tokenize_and_align_labels, batched=True)
        val_data = self.val_df.map(self.tokenize_and_align_labels, batched=True)
        
        training_args = TrainingArguments(
            output_dir=model_name,
            do_eval=True,
            lr_scheduler_type="linear",
            no_cuda=False,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            evaluation_strategy="epoch"
        )
    
        
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            
            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            results = {
                'accuracy': accuracy_score(true_labels, true_predictions),
                'f1': f1_score(true_labels, true_predictions),
                'classification_report': classification_report(true_labels, true_predictions)
            }
            return results
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=self.tokenizer, 
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()