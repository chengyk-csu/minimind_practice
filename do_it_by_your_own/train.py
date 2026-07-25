import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os

def train_epoch(train_loader,device,model,optimizer):
    model.train()
    for input_ids,labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits,loss = model(input_ids,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss={loss.item()}")


if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    vocab_size = len(tokenizer)
    model = MinimindForCausalLM(vocab_size, 100, 6, 0.1, 3, 64, 1)
    model.to(device)
    trian_dataset = PretrainDataset(train_path,tokenizer,max_seq_len=100)
    train_loader = DataLoader(train_dataset,batch_siae=4,shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
    for epoch in range(100):
        train_epoch(train_loader,device,model,optimizer)
        os.makedirs("checkpoint",exist_ok=True)
        torch.save(model.state_dict(),f"checkpoint/model_epoch{epoch}.pth")
