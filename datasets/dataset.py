import torch
import random
import pandas as pd

from torch.utils.data import Dataset

def get_csv_data(path):
    return pd.read_csv(path)

class CustomDataset(Dataset):
    def __init__(self, path, tokenizer) -> None:
        data = pd.read_csv(path)
        print('origin_data:', data.shape)

        query = []
        desc = []
        label = []
        for i, (q, d, domain) in enumerate(zip(list(data['question']), list(data['content']), list(data['Domain']))): # Domain = 주제
            if type(q) != str: continue
            if type(d) != str: continue            

            same_domain_desc_list = pd.unique(data[data['Domain']==domain]['content'])
            other_domain_desc_list =  pd.unique(data[data['Domain']!=domain]['content'])

            # Positive Pair
            query.append(q)
            desc.append(d)
            label.append(1)

            # Negative Pair
                # Hard
            negative_sample_count = 20 if len(same_domain_desc_list) >= 20 else len(same_domain_desc_list)
            desc_list_sample = random.sample(list(same_domain_desc_list), k=negative_sample_count)
            for dl in desc_list_sample:
                if d == dl: continue
                else: 
                    query.append(q)
                    desc.append(dl)
                    label.append(-1)
                
                # General
            other_desc_list_sample = random.sample(list(other_domain_desc_list), 5)
            for dl in other_desc_list_sample:
                query.append(q)
                desc.append(dl)
                label.append(-1)
            
        self.length = len(query)
        
        self.query = tokenizer(query, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
        self.desc = tokenizer(desc, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
        self.labels = torch.tensor(label)
    
    def __getitem__(self, idx):
        item = {}
        item['query'] = {'input_ids': self.query['input_ids'][idx], 'attention_mask': self.query['attention_mask'][idx]}
        item['refer'] = {'input_ids': self.desc['input_ids'][idx], 'attention_mask': self.desc['attention_mask'][idx]}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return self.length
    