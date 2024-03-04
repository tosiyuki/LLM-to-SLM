# LLM-to-SLM
A reproduced implementation of ["Think Big, Generate Quick: LLM-to-SLM for Fast Autoregressive Decoding"](https://arxiv.org/html/2402.16844v1#:~:text=LLM%2Dto%2DSLM%20can%20be,the%20LLM%20for%20prompt%20encoding.).

## Train
```
bash scripts/t5-3b-to-gpt2.sh
```

## Demo
```
python demo.py
```

## About releasing weights
- [toshi456/LLM-to-SLM-Alpaca](https://huggingface.co/toshi456/LLM-to-SLM-Alpaca)

    This model uses A as LLM and B as SLM. Training data is [Stanford Alpaca 52K](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) and cannot be used for commercial purposes.

## Acknowledgement
[Bergner et al, Think Big, Generate Quick: LLM-to-SLM for Fast Autoregressive Decoding, 2024](https://arxiv.org/html/2402.16844v1#:~:text=LLM%2Dto%2DSLM%20can%20be,the%20LLM%20for%20prompt%20encoding.)
