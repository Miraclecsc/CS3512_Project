import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def set_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--do_sample', default=True, type=bool, required=False)
    parser.add_argument('--top_k', default=8, type=int, required=False)
    parser.add_argument('--temperature', default=0.7,
                        type=float, required=False)
    parser.add_argument('--top_p', default=0.95, type=float, required=False)
    parser.add_argument('--repetition_penalty', default=1.2,
                        type=float, required=False)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument(
        '--system', default="you are a friendly AI", type=str, required=False)
    parser.add_argument(
        '--model_path', default="./TinyLlama-1.1B-Chat-v1.0-awq-111", type=str, required=False)
    # "./TinyLlama-1.1B-Chat-v1.0-AWQ" or "./TinyLlama-1.1B-Chat-v1.0"
    parser.add_argument('--is_cuda', default=True, type=bool, required=False)
    return parser.parse_args()


def generate(inp, model, tokenizer, args):
    # Using the text streamer to stream output one token at a time
    streamer = TextStreamer(tokenizer, skip_prompt=True,
                            skip_special_tokens=True)

    inp = f'''<|system|>
        {args.system}</s>
        <|user|>
        {inp}</s>
        <|assistant|>
        '''

    # Convert prompt to tokens
    tokens = tokenizer(
        inp,
        return_tensors='pt'
    ).input_ids.cuda()

    generation_params = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_len,
        "repetition_penalty": args.repetition_penalty
    }

    # Generate streamed output, visible one token at a time
    generation_output = model.generate(
        tokens,
        streamer=streamer,
        **generation_params
    )


def main():
    args = set_args()
    checkpoint = args.model_path
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        low_cpu_mem_usage=True,
        device_map=device
    )

    while True:
        try:
            print("you can chat with the model")
            inp = input()
            generate(inp, model, tokenizer, args)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
