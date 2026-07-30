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
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss={loss.item()}")

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind-3")
    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        max_seq_len=100,
        d_model=128,
        dropout=0.1,
        num_heads=4,
        num_kv_heads=4,
        dim_feedforward=512,
        num_layers=4
    )
    model = MinimindForCausalLM(config)
    model.to(device)
    train_dataset = PretrainDataset(train_path="./data/tiny_pretrain.jsonl",tokenizer=tokenizer,config=config)
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    # for epoch in range(100):
    #     train_epoch(train_loader,device,model,optimizer)
    #     os.makedirs("checkpoint",exist_ok=True)
    #     torch.save(model.state_dict(),f"checkpoint/model_epoch{epoch}.pth")
    checkpoint_path = "./checkpoint/model_epoch99.pth"

    state_dict = torch.load(checkpoint_path,map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.eval()
    prompt = "机器学习"
    inputs = tokenizer(prompt,return_tensors="pt",padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generated_ids = model.generate(batch_size=input_ids.shape[0],device=device,input_ids=input_ids,max_new_tokens=20,repetition_penalty=1.1,temperature=1.0,topk=4,top_p=0.9,do_sample=True,eos_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
    print(generated_text)
