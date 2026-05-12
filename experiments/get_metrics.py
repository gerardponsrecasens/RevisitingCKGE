import os
import json
import pandas as pd
import numpy as np

folder = "experiment_1"
pd_rows = []
for filename in os.listdir(folder):
    if filename.endswith(".json"):
        filepath = os.path.join(folder, filename)
       
        with open(filepath, "r") as f:
            data = json.load(f)
        model = data['settings']['update_technique']
        dataset = data['settings']['dataset']
        c_mrr = data['corrected']['mrr']
        c_h1 = data['corrected']['hits@1']
        c_h3 = data['corrected']['hits@3']
        c_h10 = data['corrected']['hits@10']

        if dataset != 'PS-CKGE':
            w_mrr = data['4']['mrr']
            w_h1 = data['4']['hits@1']
            w_h3 = data['4']['hits@3']
            w_h10 = data['4']['hits@10']
        else:
            w_mrr = data['2']['mrr']
            w_h1 = data['2']['hits@1']
            w_h3 = data['2']['hits@3']
            w_h10 = data['2']['hits@10']
        
        pd_rows.append([model,dataset,c_mrr,w_mrr,np.round((c_mrr-w_mrr)/w_mrr*100,1),
                        c_h1,w_h1,np.round((c_h1-w_h1)/w_h1*100,1),
                        c_h3,w_h3,np.round((c_h3-w_h3)/w_h3*100,1),
                        c_h10,w_h10,np.round((c_h10-w_h10)/w_h10*100,1)])
        
df = pd.DataFrame(pd_rows,columns=['Model','Dataset','MRR','W_MRR','%M','H@1','WH@1','%1','H@3','WH@3','%3','H@10','WH@10','%10'])

order = [
    "retraining", "finetune", "EWC", "EMR", "LKGE", "FMR",
    "ETT-CKGE", "DebiasedKGE", "FastKGE", "SAGE", "incDE"
]

df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)

df = df.sort_values("Model")


df.to_csv('corrected_metrics.csv',index=False)



folder = "experiment_1"
pd_rows = []
metric = "mrr"
for filename in os.listdir(folder):
    if filename.endswith(".json"):
        filepath = os.path.join(folder, filename)
       
        with open(filepath, "r") as f:
            data = json.load(f)
        model = data['settings']['update_technique']
        dataset = data['settings']['dataset']

        if model == 'retraining':
            continue
        h0_0 = data["0"]["local"]["0"][metric]
        if dataset != 'PS-CKGE':
            h0_4 = data["4"]["cf"]["0"]["full"][metric]
            h04_w = data["4"]["cf"]["0"]["only"][metric]
        else:
            h0_4 = data["2"]["cf"]["0"]["full"][metric]
            h04_w = data["2"]["cf"]["0"]["only"][metric]
        
        change = np.round((h0_4-h04_w)/h04_w*100,1)
        
        pd_rows.append([model,dataset,h0_0,h0_4,h04_w,change])
        
df = pd.DataFrame(pd_rows,columns=['Model','Dataset','H0_0','H0_N','WH0_N','%'])

order = [
    "retraining", "finetune", "EWC", "EMR", "LKGE", "FMR",
    "ETT-CKGE", "DebiasedKGE", "FastKGE", "SAGE", "incDE"
]

df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)

df = df.sort_values("Model")


df.to_csv('cf_sources.csv', index=False)


with open('test_sizes.json', "r") as f:
        dataset_stats = json.load(f)
        

folder = "experiment_1"
pd_rows = []
metric = 'hits@3'
for filename in os.listdir(folder):
    if filename.endswith(".json"):
        filepath = os.path.join(folder, filename)
       
        with open(filepath, "r") as f:
            data = json.load(f)
        model = data['settings']['update_technique']
        dataset = data['settings']['dataset']
        test_size = sum(dataset_stats[dataset])-dataset_stats[dataset][-1]
        n = len(dataset_stats[dataset])-1
        
        cf = 0
        cf_bad = 0
        

        if model == 'retraining':
            continue
        
        
        for i in range(n):
            size = dataset_stats[dataset][i]
            initial = data[str(i)]['cf'][str(i)]['only'][metric]
            final = data[str(n)]['cf'][str(i)]['full'][metric]
            cf +=(final-initial)*(1/initial)*(size/test_size)*(1/(4-i))
        
        for i in range(n):
            size = dataset_stats[dataset][i]
            initial = data[str(i)]['cf'][str(i)]['only'][metric]
            final = data[str(n)]['cf'][str(i)]['only'][metric]
            cf_bad +=(final-initial)*(1/initial)*(size/test_size)*(1/(4-i))

        
        pd_rows.append([model,dataset,cf,cf_bad])
        
        
df = pd.DataFrame(pd_rows,columns=['Model','Dataset','CF','CF_BAD']).round(3)

order = [
    "retraining", "finetune", "EWC", "EMR", "LKGE", "FMR",
    "ETT-CKGE", "DebiasedKGE", "FastKGE", "SAGE", "incDE"
]

df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)

df = df.sort_values("Model")


df.to_csv('cf.csv', index=False)