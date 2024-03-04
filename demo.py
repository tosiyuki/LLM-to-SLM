import torch
import transformers

from transformers.generation.streamers import TextStreamer
from llm_to_slm.model.llm_to_slm_gpt2 import Llm2SlmGpt2ForCausalLM
from llm_to_slm.train.dataset import make_prompt


if __name__ == "__main__":
    model_path = "toshi456/LLM-to-SLM-Alpaca"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = Llm2SlmGpt2ForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    slm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "google-t5/t5-3b",
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    instruction = "Describe the structure of a gold."
    input = ""
    prompt = make_prompt(
        instruction=instruction,
        input=input
    )
    input_ids = slm_tokenizer(
        prompt, 
        return_tensors='pt',
        padding="longest",
        max_length=slm_tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0].view(1, -1)
    llm_input_ids = llm_tokenizer(
        prompt, 
        return_tensors='pt',
        padding="max_length",
        max_length=input_ids.shape[1],
        truncation=True,
    ).input_ids[0].view(1, -1)
    
    input_ids = input_ids.to(model.device)
    llm_input_ids = llm_input_ids.to(model.device)

    streamer = TextStreamer(slm_tokenizer, skip_prompt=True, timeout=20.0)

    # predict
    with torch.inference_mode():
        model.generate(
            inputs=input_ids,
            llm_input_ids=llm_input_ids,
            do_sample=False,
            max_new_tokens=256,
            streamer=streamer,
            use_cache=True
        )