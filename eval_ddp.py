import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from math_evaluator import is_correct_minerva_with_doubao
from torch.utils.data import Dataset, DataLoader, Sampler
from utils import tokenize_and_postprocess_data
from template import QUERY_TEMPLATE, ANSWER_PATTERN
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import math
from typing import TypeVar, Optional, Iterator

T_co = TypeVar('T_co', covariant=True)

def check_answer(decoded_prompt, decoded_response, golden_answer, logger):
    correct_solution_score = 1.0
    has_answer_score = 0.1
    answer = re.search(ANSWER_PATTERN, decoded_response)
    if answer is None:
        log = []
        log.append(f"DECODED PROMPT = {decoded_prompt}")
        log.append(f"DECODED RESPONSE = {decoded_response}")
        log.append(f"GOLDEN ANSWER = {golden_answer}")
        log.append(f"*** NO ANSWER *** Assigning 0 point to the response")
        # logger(f"\n".join(log))
        score = 0.
    else:
        final_answer = answer.group(0)
        if is_correct_minerva_with_doubao(decoded_prompt, final_answer, golden_answer):
            score = correct_solution_score
        else:
            score = has_answer_score

        log = []
        log.append(f"DECODED PROMPT = {decoded_prompt}")
        log.append(f"DECODED RESPONSE = {decoded_response}")
        log.append(f"GOLDEN ANSWER = {golden_answer}")
        log.append(f"MODEL ANSWER = {final_answer}")
        if score == correct_solution_score:
            log.append(f"*** CORRECT *** Assigning {correct_solution_score} point(s) to the response.")
            # logger(f"\n".join(log))
        else:
            log.append(f"*** INCORRECT *** Assigning {has_answer_score} point(s) to the response.")
            # logger(f"\n".join(log))
    return score == correct_solution_score

class MATHDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.prompts = []
        self.answers = []
        self.tokenizer = tokenizer
        for item in data:
            problem = item['problem']
            answer = item['answer']
            prompt = QUERY_TEMPLATE.format(Question=problem)
            self.prompts.append(prompt)
            self.answers.append(answer)

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        input = {}
        prompt = self.prompts[index]
        answer = self.answers[index]
        input_ids, attention_mask = tokenize_and_postprocess_data(prompt, self.tokenizer, max_length=512, pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation='right')
        input['input_ids'] = input_ids[0]
        input['attention_mask'] = attention_mask[0]
        return input, answer

class DistributedValidationSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        return math.ceil((len(self.dataset) - self.rank) / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

class Logger:
    def __init__(self, rank):
        self.rank = rank

    def __call__(self, message):
        print(f"[Rank {self.rank}] {message}")

def main(model_name, data_path):
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    logger = Logger(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model = DDP(model, device_ids=[local_rank])

    dataset = load_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = MATHDataset(dataset['test'], tokenizer)
    sampler = DistributedValidationSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, sampler=sampler)
    logger(f"batch size: {test_dataloader.batch_size}")
    logger(f"length of test dataloader: {len(test_dataloader)}")

    dist.barrier()

    correct = torch.tensor(0, dtype=torch.int64, device=device)
    total = torch.tensor(0, dtype=torch.int64, device=device)
    accuracy = 0
    
    for inputs, answers in test_dataloader:
        outputs = model.module.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_new_tokens=2048,
            pad_token_id = tokenizer.eos_token_id
        )

        for i in tqdm(range(len(inputs['input_ids'])), desc=f"[Rank {local_rank}] Verify Loop"):
            golden_answer = answers[i]
            decoded_prompt = tokenizer.decode(inputs['input_ids'][i])

            model_response = outputs[i][len(inputs['input_ids'][i]):]
            decoded_response = tokenizer.decode(model_response)

            if check_answer(decoded_prompt, decoded_response, golden_answer, logger):
                correct += 1
            total += 1

        old_correct = correct.clone()
        old_total = total.clone()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        accuracy = correct.item() / total.item()
        if local_rank == 0:
            logger(f"Correct/Total: {correct.item()} / {total.item()}, Accuracy: {accuracy:.2%}")
        correct = old_correct
        total = old_total

if __name__ == "__main__":
    model_name = ""
    data_path = ""
    main()