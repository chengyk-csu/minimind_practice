import torch
from transformers import AutoTokenizer
from model import MinimindForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./checkpoints_sft/epoch_3"
tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind-3")
model = MinimindForCausalLM.from_pretrained(model_path).to(device)

model.eval()
conversations = []
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    conversations.append(
        {
            "role": "user",
            "content": user_input
        }
    )
    prompt = tokenizer.apply_chat_template(conversations,tokenize=False,add_generation_prompt=True)
    inputs = tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_length = input_ids.shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            batch_size=input_ids.shape[0],
            device=device,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            temperature=0.8,
            topk=50,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    new_token_ids = generated_ids[:,prompt_length:]
    response = tokenizer.decode(new_token_ids[0],skip_special_tokens=True)
    print(f"Assistant: {response}")
    conversations.append(
        {
            "role": "assistant",
            "content": response
        }
    )
