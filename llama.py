import os
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import Optional, Any, Dict, List

llm: Optional[LlamaCpp] = None
callback_manager: Any = None

model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
template = """
    {system_prompt}</s>

    {question}</s>
    """

default_prompt = "You are a smart and friendly AI assistant. You are here to help me with my questions."

with open("data.json", "r") as f:
    data = json.load(f)

data_system_prompt = data["system_prompt"]
data_question = data["question"]
data_response = data["response"]


class StreamingCustomCallbackHandler(StreamingStdOutCallbackHandler):
    """ Callback handler for LLM streaming """

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """ Run when LLM starts running """
        print("<LLM Started>")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """ Run when LLM ends running """
        print("<LLM Ended>")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """ Run on new LLM token. Only available when streaming is enabled """
        print(f"{token}", end="")


def llm_init():
    """ Load large language model """
    global llm, callback_manager

    callback_manager = CallbackManager([StreamingCustomCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_file,
        temperature=0.1,
        n_gpu_layers=0,
        n_batch=256,
        callback_manager=callback_manager,
        verbose=False
    )


def generate_response(system_prompt, question):
    """ Generate response from the model """
    max_length = 512
    global llm, template

    if not system_prompt:
        system_prompt = default_prompt

    inp = template.format(system_prompt=system_prompt, question=question)
    inp = inp[:max_length]
    prompt_template = PromptTemplate(template=inp)

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke(
        {"system_prompt": system_prompt, "question": question}, config={})

    return response


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
    llm_init()

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
            generated_text = generate_response(
                data_system_prompt[i], data_question[i])
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
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            continue

    if count > 0:
        avg_scores = {
            "Average BLEU Score": total_bleu / count,
            "Average ROUGE-1 Score": total_rouge1 / count,
            "Average ROUGE-2 Score": total_rouge2 / count,
            "Average ROUGE-L Score": total_rougeL / count
        }
        results.append(avg_scores)

    with open("llama_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
