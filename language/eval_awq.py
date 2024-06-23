import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm

default_prompt = "You are a smart and friendly AI assistant. You are here to help me with my questions."

with open("data.json", "r") as f:
    data = json.load(f)

data_system_prompt = data["system_prompt"]
data_question = data["question"]
data_response = data["response"]


def set_args():
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument('--do_sample', default=True, type=bool, required=False)
    parser.add_argument('--top_k', default=8, type=int, required=False)
    parser.add_argument('--temperature', default=0.7,
                        type=float, required=False)
    parser.add_argument('--top_p', default=0.95, type=float, required=False)
    parser.add_argument('--repetition_penalty', default=1.2,
                        type=float, required=False)
    parser.add_argument('--max_len', type=int, default=512)

    # other settings
    parser.add_argument(
        '--model_path', default="./TinyLlama-1.1B-Chat-v1.0-awq-111", type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true',
                        help="Use this flag to disable CUDA")

    return parser.parse_args()


def generate(system_prompt, question, model, tokenizer, args, device):
    # streamer = TextStreamer(tokenizer, skip_prompt=True,
    #                         skip_special_tokens=True)

    if system_prompt is None or system_prompt == "":
        system_prompt = default_prompt

    inp = f'''
    {system_prompt}</s>

    {question}</s>
    '''

    tokens = tokenizer(inp, return_tensors='pt').input_ids.to(device)

    generation_params = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_len,
        "repetition_penalty": args.repetition_penalty
    }

    generation_output = model.generate(
        tokens, **generation_params)
    generated_text = tokenizer.decode(
        generation_output[0], skip_special_tokens=True)

    return generated_text


def evaluate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis)


def evaluate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


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
        checkpoint, low_cpu_mem_usage=True, device_map=device)

    total_bleu = 0
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    count = 0

    results = []

    for i in tqdm(range(len(data_system_prompt))):
        try:
            result = {}
            result["Prompt"] = data_system_prompt[i]
            result["Question"] = data_question[i]
            result["Ground Truth"] = data_response[i]
            generated_text = generate(
                data_system_prompt[i], data_question[i], model, tokenizer, args, device)
            result["Generated Response"] = generated_text

            bleu_score = evaluate_bleu(data_response[i], generated_text)
            rouge_scores = evaluate_rouge(data_response[i], generated_text)

            result["BLEU Score"] = bleu_score
            result["ROUGE Scores"] = {
                "ROUGE-1": rouge_scores['rouge1'].fmeasure,
                "ROUGE-2": rouge_scores['rouge2'].fmeasure,
                "ROUGE-L": rouge_scores['rougeL'].fmeasure
            }

            total_bleu += bleu_score
            total_rouge1 += rouge_scores['rouge1'].fmeasure
            total_rouge2 += rouge_scores['rouge2'].fmeasure
            total_rougeL += rouge_scores['rougeL'].fmeasure
            count += 1

            results.append(result)
        except KeyboardInterrupt:
            break

    if count > 0:
        avg_bleu = total_bleu / count
        avg_rouge1 = total_rouge1 / count
        avg_rouge2 = total_rouge2 / count
        avg_rougeL = total_rougeL / count

        print(f"Average BLEU Score: {avg_bleu}")
        print(f"Average ROUGE-1 Score: {avg_rouge1}")
        print(f"Average ROUGE-2 Score: {avg_rouge2}")
        print(f"Average ROUGE-L Score: {avg_rougeL}")

        avg_scores = {
            "Average BLEU Score": avg_bleu,
            "Average ROUGE-1 Score": avg_rouge1,
            "Average ROUGE-2 Score": avg_rouge2,
            "Average ROUGE-L Score": avg_rougeL
        }
        results.append(avg_scores)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
