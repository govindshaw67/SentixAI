"""Train controlled SentixAI model and hyperparameter experiments; save report evidence."""

from __future__ import annotations

import argparse, copy, csv, json, random, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam, AdamW, RMSprop
from torch.utils.data import DataLoader

from comparison_models import MODELS
from sentixai_data import SentimentDataset, build_vocabulary, load_and_split_data


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def select_device(name):
    return torch.device("mps" if name == "auto" and torch.backends.mps.is_available() else ("cpu" if name == "auto" else name))

def metrics(y, pred):
    weighted = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    macro = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy":float(accuracy_score(y,pred)), "precision":float(weighted[0]), "recall":float(weighted[1]), "f1":float(weighted[2]), "macro_f1":float(macro[2])}

def epoch(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None); total=0.; ys=[]; ps=[]
    context=torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if optimizer: optimizer.zero_grad(set_to_none=True)
            logits,_=model(x); loss=loss_fn(logits,y)
            if optimizer:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total += loss.item()*len(y); ys += y.cpu().tolist(); ps += logits.argmax(1).cpu().tolist()
    return total/len(loader.dataset), ys, ps

def make_optimizer(name, params, lr, weight_decay):
    return {"adam":Adam, "adamw":AdamW, "rmsprop":RMSprop}[name.lower()](params, lr=lr, weight_decay=weight_decay)

def plot_confusion(y,pred,names,path,title):
    fig,ax=plt.subplots(figsize=(5,4)); sns.heatmap(confusion_matrix(y,pred),annot=True,fmt="d",cmap="Blues",xticklabels=names,yticklabels=names,ax=ax); ax.set(xlabel="Predicted",ylabel="Actual",title=title);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def train_one(name, spec, data, vocab, loaders, class_names, root, seed, device):
    seed_everything(seed); out=root/name; out.mkdir(parents=True,exist_ok=True)
    cls=MODELS[spec["model"]]; cfg={k:spec[k] for k in ("embedding_dim","hidden_dim","dropout")}
    model=cls(len(vocab),len(class_names),padding_idx=vocab["<PAD>"],**cfg).to(device)
    optimizer=make_optimizer(spec["optimizer"],model.parameters(),spec["learning_rate"],spec["weight_decay"]); loss_fn=nn.CrossEntropyLoss()
    history={"train_loss":[],"validation_loss":[],"train_accuracy":[],"validation_accuracy":[]}; best=None; best_loss=float("inf"); patience=spec["patience"]; start=time.perf_counter()
    for idx in range(spec["epochs"]):
        tl,ty,tp=epoch(model,loaders[0],loss_fn,device,optimizer); vl,vy,vp=epoch(model,loaders[1],loss_fn,device)
        history["train_loss"].append(tl);history["validation_loss"].append(vl);history["train_accuracy"].append(metrics(ty,tp)["accuracy"]);history["validation_accuracy"].append(metrics(vy,vp)["accuracy"])
        print(f"{name} epoch {idx+1}/{spec['epochs']} | val loss {vl:.4f} | val macro-F1 {metrics(vy,vp)['macro_f1']:.4f}")
        if vl < best_loss: best_loss=vl; best=copy.deepcopy(model.state_dict()); patience=spec["patience"]
        else:
            patience-=1
            if patience==0: print(f"{name}: early stopping"); break
    elapsed=time.perf_counter()-start; model.load_state_dict(best); test_loss,y,pred=epoch(model,loaders[2],loss_fn,device); m=metrics(y,pred)
    report=classification_report(y,pred,target_names=class_names,zero_division=0,output_dict=True); params=sum(p.numel() for p in model.parameters()); checkpoint=out/"model.pt"; torch.save({"model":spec["model"],"config":cfg,"state_dict":model.state_dict(),"class_names":class_names},checkpoint)
    size=checkpoint.stat().st_size; plot_confusion(y,pred,class_names,out/"confusion_matrix.png",name)
    record={"name":name,"model":spec["model"],**{k:spec[k] for k in ("learning_rate","batch_size","epochs","dropout","hidden_dim","embedding_dim","optimizer","weight_decay")},**m,"test_loss":test_loss,"training_seconds":elapsed,"parameters":params,"model_size_bytes":size,"epochs_completed":len(history["train_loss"]),"history":history,"classification_report":report}
    (out/"result.json").write_text(json.dumps(record,indent=2)); return record

def plot_summary(records, root, filename, title):
    labels=[r["name"] for r in records]; fig,ax=plt.subplots(figsize=(max(9,len(labels)*1.7),5)); x=np.arange(len(labels)); ax.bar(x-.18,[r["accuracy"] for r in records],.36,label="Accuracy");ax.bar(x+.18,[r["macro_f1"] for r in records],.36,label="Macro-F1");ax.set_xticks(x,labels,rotation=20,ha="right");ax.set_ylim(0,1);ax.legend();ax.set_title(title);fig.tight_layout();fig.savefig(root/filename,dpi=170);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="experiment_config.json");parser.add_argument("--csv",help="Override the dataset path in the config.");parser.add_argument("--output-dir",help="Override the output directory in the config.");parser.add_argument("--device",choices=["cpu","mps","auto"],help="Override the compute device in the config.");args=parser.parse_args();config=json.loads(Path(args.config).read_text());config["dataset"]=args.csv or config["dataset"];config["output_dir"]=args.output_dir or config["output_dir"];config["device"]=args.device or config["device"];dataset_path=Path(config["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Pass --csv with the IMDB CSV path.")
    seed_everything(config["seed"]);root=Path(config["output_dir"]);root.mkdir(parents=True,exist_ok=True);shared=config["shared"]
    data=load_and_split_data(config["dataset"],config["seed"]);vocab=build_vocabulary(data.train_tokens,shared["min_frequency"],shared["max_vocab_size"]);datasets=[SentimentDataset(t,l,vocab,shared["max_length"]) for t,l in ((data.train_tokens,data.train_labels),(data.validation_tokens,data.validation_labels),(data.test_tokens,data.test_labels))]
    device=select_device(config["device"]); print(f"Device: {device}; train/validation/test: {[len(x) for x in datasets]}; vocab: {len(vocab)}")
    records=[];runs=[]
    for model in config["model_comparison"]: runs.append((f"comparison_{model}",{"model":model}))
    for experiment in config["hyperparameter_experiments"]: runs.append((experiment["name"],experiment))
    for name, override in runs:
        spec={**shared,**override}; gen=torch.Generator().manual_seed(config["seed"]);loaders=(DataLoader(datasets[0],batch_size=spec["batch_size"],shuffle=True,generator=gen),DataLoader(datasets[1],batch_size=spec["batch_size"]),DataLoader(datasets[2],batch_size=spec["batch_size"]))
        records.append(train_one(name,spec,data,vocab,loaders,data.class_names,root,config["seed"],device))
    best=max(records,key=lambda r:r["macro_f1"]); fields=["name","model","accuracy","precision","recall","f1","macro_f1","training_seconds","parameters","model_size_bytes","learning_rate","batch_size","epochs","dropout","hidden_dim","embedding_dim","optimizer"]
    with (root/"comparison_results.csv").open("w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows([{k:r[k] for k in fields} for r in records])
    summary={"selection_metric":"macro_f1","best_experiment":best,"all_results":records,"dataset":{"classes":data.class_names,"splits":{"train":len(datasets[0]),"validation":len(datasets[1]),"test":len(datasets[2])},"vocabulary_size":len(vocab),"device":str(device)}};(root/"experiment_summary.json").write_text(json.dumps(summary,indent=2));plot_summary([r for r in records if r["name"].startswith("comparison_")],root,"model_comparison_chart.png","Architecture comparison");plot_summary([r for r in records if not r["name"].startswith("comparison_")],root,"hyperparameter_chart.png","Hyperparameter experiments");print("\nBEST CONFIGURATION (macro-F1):\n"+json.dumps({k:best[k] for k in fields},indent=2))
if __name__=="__main__": main()
