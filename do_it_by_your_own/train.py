import torch
import math
from contextlib import nullcontext
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import MiniMindConfig, MinimindForCausalLM
from dataset import PretrainDataset
import os

def save_checkpoint(model,optimizer,scheduler,epoch,global_step,save_dir):
    checkpoint_dir = os.path.join(save_dir,f"epoch_{epoch + 1}")
    os.makedirs(checkpoint_dir,exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        },os.path.join(checkpoint_dir,"trainer_state.pt")
    )

def train_epoch(model,dataloader,optimizer,scheduler,device,global_step,gradient_accumulation_steps=4,max_grad_norm=1.0,log_interval=20):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
        group_end = min(group_start + gradient_accumulation_steps,len(dataloader))
        current_group_size = (group_end - group_start)
        use_amp = (device.type == "cuda"and torch.cuda.is_bf16_supported())
        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            )if use_amp
            else nullcontext()
        )
        with amp_context:
            outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,use_cache=False)
            lm_loss = outputs.loss
            aux_loss = outputs.aux_loss
            loss = lm_loss + aux_loss
            scaled_loss = (loss / current_group_size)
        scaled_loss.backward()
        total_loss += loss.item()
        should_update = (step + 1 == group_end)
        if should_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            optimizer.step()
            global_step += 1
            if global_step % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"step={global_step}, "
                    f"loss={loss.item():.4f}, "
                    f"lm_loss={lm_loss.item():.4f}, "
                    f"aux_loss={aux_loss.item():.6f}, "
                    f"lr={current_lr:.6e}"
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    average_loss = total_loss / len(dataloader)
    return average_loss,global_step

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
    num_epochs = 3
    learning_rate = 3e-4
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    warmup_ratio = 0.05
    save_dir = "./checkpoints"
    resume_from_checkpoint = None
    log_interval = 20
    if resume_from_checkpoint is not None:
        model = MinimindForCausalLM.from_pretrained(resume_from_checkpoint).to(device)
    else:
        model = MinimindForCausalLM(config).to(device)
    train_dataset = PretrainDataset(train_path="./data/tiny_pretrain.jsonl",tokenizer=tokenizer,config=config)
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    num_training_steps = updates_per_epoch * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    start_epoch = 0
    global_step = 0
    if resume_from_checkpoint is not None:
        trainer_state = torch.load(os.path.join(resume_from_checkpoint,"trainer_state.pt"),
            map_location=device
        )
        optimizer.load_state_dict(trainer_state["optimizer"])
        scheduler.load_state_dict(trainer_state["scheduler"])
        start_epoch = (trainer_state["epoch"] + 1)
        global_step = trainer_state["global_step"]
    for epoch in range(start_epoch,num_epochs):
        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            global_step=global_step,
            log_interval=log_interval
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"avg_loss={avg_loss:.4f}"
        )
        save_checkpoint(model=model,optimizer=optimizer,scheduler=scheduler,epoch=epoch,global_step=global_step,save_dir=save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    model = MinimindForCausalLM.from_pretrained(save_dir).to(device)

    model.eval()
    prompt = "机器学习"
    inputs = tokenizer(prompt,return_tensors="pt",padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    generated_ids = model.generate(batch_size=input_ids.shape[0],device=device,input_ids=input_ids,max_new_tokens=20,repetition_penalty=1.1,temperature=1.0,topk=4,top_p=0.9,do_sample=True,eos_token_id=tokenizer.eos_token_id,attention_mask=attention_mask)
    generated_text = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
    print(generated_text)
