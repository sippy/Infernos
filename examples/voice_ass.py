import torch
from transformers import AutoTokenizer, AutoConfig
from ipex_llm.transformers import AutoModelForCausalLM
from datetime import datetime

model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
#config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
#local_cache = f"~/.cache/Infernos/{model_name}"
#config.save_pretrained(local_cache)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",
device_map="auto",
             load_in_4bit=True,
             optimize_model=True,
             trust_remote_code=True,
             use_cache=True
         )
#model = model.half().to("xpu")
model = model.to("xpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful voice auto-attendant for the company Sippy Software. Start by greeting the caller and asking how you can help. Try to keep your messages brief and concise to reduce latency."}, {"role": "system", "content": f'<Now is {datetime.now()}> <Incoming call from "Doe Joe" +11233742223>'}]
text = tokenizer.apply_chat_template(messages,
            tokenize=False,
            add_generation_prompt=True
        )
for i in range(10):
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=16 * 1024, output_scores=True, return_dict_in_generate=True)
    torch.xpu.synchronize()
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids.sequences)
        ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(messages, response)
