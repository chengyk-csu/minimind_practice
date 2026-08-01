import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from model import MinimindForCausalLM
from dataset import SFTDataset
from train import train_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_checkpoint = "./checkpoints/epoch_3"
sft_data_path = "./data/sft_data.jsonl"
sft_save_dir = "./checkpoints_sft"
num_epochs = 3
batch_size = 2
max_seq_len = 512
learning_rate = 1e-5
gradient_accumulation_steps = 4
max_grad_norm = 1.0
warmup_ratio = 0.05

tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind-3")
dataset = SFTDataset(data_path=sft_data_path,tokenizer=tokenizer,max_seq_len=max_seq_len)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
model = MinimindForCausalLM.from_pretrained(pretrain_checkpoint).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
updates_per_epoch = math.ceil(len(dataloader)/ gradient_accumulation_steps)
num_training_steps = (updates_per_epoch * num_epochs)
num_warmup_steps = int(num_training_steps * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
# 修复：SFT 训练也需要初始化并持续传递全局更新步数。
global_step = 0
for epoch in range(num_epochs):
    avg_loss, global_step = train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        global_step=global_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm
    )
    print(
        f"SFT Epoch {epoch + 1}/{num_epochs}, "
        f"loss={avg_loss:.4f}"
    )
    model.save_pretrained(
        f"{sft_save_dir}/epoch_{epoch + 1}"
    )
