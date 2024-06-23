import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
# from asr import transcribe_file, transcribe_mic
# from datasets import load_dataset

default_prompt = "You are a smart and friendly AI."

# data = load_dataset("Open-Orca/OpenOrca")
# data = data["train"][:200]
# # save
# with open("data.json", "w") as f:
#     json.dump(data, f)

with open("data.json", "r") as f:
    data = json.load(f)

data_system_prompt = data["system_prompt"]
data_question = data["question"]
data_response = data["response"]

print(data_system_prompt[1], '\n')
print(data_question[1], '\n')
print(data_response[1], '\n')


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
    parser.add_argument('--system', default=default_prompt,
                        type=str, required=False)
    parser.add_argument(
        '--model_path', default="./TinyLlama-1.1B-Chat-v1.0-awq-111", type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true',
                        help="Use this flag to disable CUDA")

    return parser.parse_args()


def generate(inp, model, tokenizer, args, device):
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
    tokens = tokenizer(inp, return_tensors='pt').input_ids.to(device)

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
        tokens, streamer=streamer, **generation_params)


def main():
    args = set_args()
    checkpoint = args.model_path

    if not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        device = "cpu"
    else:
        device = "cpu" if args.no_cuda else "cuda"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, device_map=device
    )

    audio_file = "sample2.flac"
    if audio_file:
        print(f"Transcribing audio file: {audio_file}")
        inp = transcribe_file(audio_file)
        print(f"Input: ")
        print(inp)
        print("Output:")
        generate(inp, model, tokenizer, args, device)
    else:
        while True:
            try:
                print("you can chat with the model")
                inp = input()
                generate(inp, model, tokenizer, args, device)
            except KeyboardInterrupt:
                break


if __name__ == '__main__':
    main()
