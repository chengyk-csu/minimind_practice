import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import MiniMindConfig, MinimindForCausalLM
from dataset import PretrainDataset
import os

def train_epoch(train_loader,device,model,optimizer):
    model.train()
    #for input_ids,labels in train_loader:
    for batch_idx,(input_ids,attention_mask,labels) in enumerate(train_loader):
        attention_mask = attention_mask.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids,attention_mask=attention_mask,labels=labels)
        lm_loss = outputs.loss
        aux_loss = outputs.aux_loss
        loss = lm_loss+aux_loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss={loss.item()}")

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind-3")
    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        max_seq_len=512,
        d_model=128,
        dropout=0.1,
        num_heads=8,
        num_kv_heads=4,
        dim_feedforward=512,
        num_layers=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    model = MinimindForCausalLM(config)
    model.to(device)
    train_dataset = PretrainDataset(train_path="./data/tiny_pretrain.jsonl",tokenizer=tokenizer,config=config)
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    save_directory = "./checkpoint"
    for epoch in range(100):
        train_epoch(train_loader,device,model,optimizer)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    model = MinimindForCausalLM.from_pretrained(save_directory).to(device)

    model.eval()
    prompt = "机器学习"
    inputs = tokenizer(prompt,return_tensors="pt",padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generated_ids = model.generate(batch_size=input_ids.shape[0],device=device,input_ids=input_ids,max_new_tokens=20,repetition_penalty=1.1,temperature=1.0,topk=4,top_p=0.9,do_sample=True,eos_token_id=tokenizer.eos_token_id,attention_mask=attention_mask)
    generated_text = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
    print(generated_text)
