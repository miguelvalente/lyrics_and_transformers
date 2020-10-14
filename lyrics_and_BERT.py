import os
import wandb
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer, BertTokenizer
from transformers import DistilBertForSequenceClassification ,BertForSequenceClassification



def main(args):

    #arguments passed 
    lyrics_path = args.lyrics_path
    batch_size = args.batch_size
    epochs = args.epochs
    model_choice = args.model_choice
    wandb_project_name = args.wandb_project_name

    models_path = "models"
    try:
        os.mkdir(models_path)
    except OSError:
        print ("Creation of the directory %s failed" % models_path)
    else:
        print ("Successfully created the directory %s " % models_path)

    if model_choice:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    with open(lyrics_path, "rb") as pkl_file:
        df = pd.read_pickle(pkl_file)    

    #This Block do the balancing od data. The number of samples on each class depends on the class with the fewest samples
    g = df.groupby('genre')
    df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    
    #This Block gives an id to each sample. Creates a dictionary where each genre gets a corresponding numerical label
    id_list = [i for i in range(len(df))]
    df['input_ids'] = id_list
    df.set_index('input_ids', inplace=True)
    print(df["genre"].value_counts())
    possible_labels = df.genre.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['label'] = df.genre.replace(label_dict)

    #splits the data and atributes a data_type label to make it easer to select the data when Encoding it
    print("Splitting Data...")
    X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                    df.label.values, 
                                                    test_size=0.15, 
                                                    random_state=36)
    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    print(df.groupby(['genre', 'label', 'data_type']).count())
    
    print("Defining Encoders...")
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt',
    )
    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].text.values, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        pad_to_max_length=True, 
        max_length=512, 
        return_tensors='pt',
    )
    print("Datasets creation...")
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    if model_choice:
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)
    else:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(label_dict),
                                                        output_attentions=False,
                                                        output_hidden_states=False)
        
    print("Dataloaders creation...")
    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)
    optimizer = AdamW(model.parameters(),
                    lr=1e-5, 
                    eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train)*epochs)
        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"This model is using device:{device} to compute")
    
    wandb.init(
        project=wandb_project_name,
        config={"epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": 1e-5,
                "max_lenght": 512}
    )

    wandb.run.save()
    run_name = wandb.run.name
    wandb.watch(model)

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch_idx, batch in enumerate(progress_bar):
  
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
           
 
        torch.save(model.state_dict(),  f'models/lyrics_{run_name}_epoch_{epoch}')
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        
        wandb.log({"Trainning Loss": loss_train_avg,
                "Validation Loss": val_loss,
                "F1 Score": val_f1})
    

    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(preds, labels):    
        label_dict_inverse = {v: k for k, v in label_dict.items()}
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

    def evaluate(dataloader_val):
        model.eval()
        
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in dataloader_val:
            if torch.cuda.is_available():
                batch = tuple(b.to(device) for b in batch)
            
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'labels':         batch[2],
                        }
                with torch.no_grad():        
                    outputs = model(**inputs)
                    
                loss = outputs[0]
                logits = outputs[1]
                loss_val_total += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = inputs['labels'].cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)
            
        loss_val_avg = loss_val_total/len(dataloader_val) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
                
        return loss_val_avg, predictions, true_vals


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--lyrics-path",nargs='?', const="data/clean_lirics.pkl", type=str, default=1, required = False,help="Path of lirics dataframe")
    parser.add_argument("--batch_size",nargs='?', const=2, type=int, default=1, required = False,help="Number of batches")
    parser.add_argument("--epochs",nargs='?', const=2, type=int, default=1, required = False,help="Number of epochs")
    parser.add_argument("--model_choice",nargs='?', const=1, type=int, default=1, required = False,help="The model you use. Eg: BERT, DistilBERT")
    parser.add_argument("--wandb_project_name",nargs='?', const="project_bert", type=str, default=1, required = False,help="The project name you use")

    args = parser.parse_args()
    main(args)