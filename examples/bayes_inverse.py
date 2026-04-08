import os
import pdb
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pandas as pd
from dotenv import load_dotenv
from einops import rearrange
from tqdm import tqdm

from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP, prepare_subset
from model import LlamaModel
from utils.weight_utils import load_model_weights
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt
from utils.logger import avg_logger, avg_acc_logger


METHOD_SET = ["zero_shot", "naive_prompting", "full_finetune"]


def is_required_training(method):
    assert method in METHOD_SET
    return method == "full_finetune"


def get_seq_log_prob(prompts, tokenizer, model, device):
    if isinstance(prompts, str):
        prompts = [prompts]
    encoded = tokenizer.encode(prompts, return_tensors="pt", return_attention_mask=True)
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    log_prob, _ = model(input_ids=ids, attention_mask=mask)

    sl = log_prob[:, :-1, :]
    ids_shift = ids[:, 1:]
    mask_shift = mask[:, 1:]

    gathered = sl.gather(-1, ids_shift.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask_shift
    return gathered.sum(dim=-1)


def bayes_inverse_llm_classifier(args, model, batch, tokenizer, device):
    _, subjects, messages, labels = batch
    subjects = [str(s) for s in subjects]
    messages = [str(m) for m in messages]

    prompts_ham = [
        get_prompt(subj, msg, ENRON_LABEL_INDEX_MAP.inv[0], args.max_seq_len, args.user_prompt)
        for subj, msg in zip(subjects, messages)
    ]
    prompts_spam = [
        get_prompt(subj, msg, ENRON_LABEL_INDEX_MAP.inv[1], args.max_seq_len, args.user_prompt)
        for subj, msg in zip(subjects, messages)
    ]

    prompts = prompts_ham + prompts_spam

    old_mode = model.training
    model.eval()

    with torch.no_grad():
        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)
        seq_log_prob = rearrange(seq_log_prob, "(c b) -> b c", c=2)
        probs = F.softmax(seq_log_prob, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        labels_tensor = torch.as_tensor(labels)

        if (labels_tensor == -1).any():
            is_correct = None
        else:
            is_correct = (preds.cpu() == labels_tensor)

    if old_mode:
        model.train()

    return is_correct, (probs.cpu(), preds.cpu())


def train_or_test(args, model, tokenizer, batch, optimizer=None, is_training=True, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.train() if is_training else model.eval()

    _, subjects, messages, label_idx = batch
    labels_tensor = torch.as_tensor(label_idx)

    if (labels_tensor == -1).any():
        bpd = None
    else:
        subjects = [str(s) for s in subjects]
        messages = [str(m) for m in messages]
        labels_text = [ENRON_LABEL_INDEX_MAP.inv[int(i)] for i in label_idx]

        prompts = [
            get_prompt(subj, msg, label, args.max_seq_len, args.user_prompt)
            for subj, msg, label in zip(subjects, messages, labels_text)
        ]

        seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)
        nchar = torch.tensor([len(p) for p in prompts], device=device).sum()
        bpd = -seq_log_prob.sum() / nchar

        if is_training:
            optimizer.zero_grad()
            bpd.backward()
            optimizer.step()

    is_correct, rest = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device)
    return bpd, is_correct, rest


def save_probs(args, model, tokenizer, dataloader, device, name="test"):
    out = os.path.join(os.getcwd(), f"{args.prob_output_folder}/{name}_dataset_probs.csv")
    if os.path.exists(out):
        os.remove(out)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"saving probabilities ({name})"):
            _, (probs, _) = bayes_inverse_llm_classifier(args, model, batch, tokenizer, device)

            idxs = torch.as_tensor(batch[0]).view(-1).tolist()
            rows = zip(idxs, probs[:, 0].tolist(), probs[:, 1].tolist())

            exists = os.path.exists(out)
            with open(out, "a") as f:
                if not exists:
                    f.write("data_index,prob_ham,prob_spam\n")
                f.writelines(f"{i},{h},{s}\n" for i, h, s in rows)


def average_ensemble_csvs(folder, base, K):
    dfs = []
    for k in range(K):
        path = os.path.join(folder, f"{base}_model{k}_dataset_probs.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path).sort_values("data_index").reset_index(drop=True)
        dfs.append(df)

    base_df = dfs[0][["data_index"]].copy()
    base_df["prob_ham"] = sum(df["prob_ham"] for df in dfs) / K
    base_df["prob_spam"] = sum(df["prob_spam"] for df in dfs) / K

    out = os.path.join(folder, f"{base}_dataset_probs.csv")
    base_df.to_csv(out, index=False)


def save_checkpoint(model, config, args, k, ckpt_dir="checkpoints"):
    os.makedirs(ckpt_dir, exist_ok=True)
    p = os.path.join(ckpt_dir, f"model_ens{k}.pt")
    torch.save({"model_state": model.state_dict(),
                "config": config,
                "args": vars(args),
                "ensemble_idx": k},
               p)
    print(f"Saved checkpoint: {p}")


if __name__ == "__main__":
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="zero_shot", choices=METHOD_SET)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dataset_path", type=str,
                        default="autograder/cpen455_released_datasets/train_val_subset.csv")
    parser.add_argument("--test_dataset_path", type=str,
                        default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--user_prompt", type=str, default="")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_ensembles", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    checkpoint = os.getenv("MODEL_CHECKPOINT")
    cache = os.getenv("MODEL_CACHE_DIR")
    device = set_device()

    tokenizer = Tokenizer.from_pretrained(checkpoint, cache_dir=cache)
    base_path = _resolve_snapshot_path(checkpoint, cache_dir=cache)
    config = Config._find_config_files(base_path)

    # data
    full = CPEN455_2025_W1_Dataset(csv_path=args.dataset_path)
    train, val = prepare_subset(full, len(full), ratio_spam=0.5, return_remaining=True)
    test = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    full_loader = DataLoader(full, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.prob_output_folder, exist_ok=True)

    # ensemble
    for k in range(args.num_ensembles):
        torch.manual_seed(k)

        model = LlamaModel(config)
        load_model_weights(model, checkpoint, cache_dir=cache, device=device)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        for it in tqdm(range(args.num_iterations), desc=f"Training (ens {k})"):
            if (it + 1) % 10 == 0:
                acc_log = avg_acc_logger()
                bpd_log = avg_logger()
                with torch.no_grad():
                    for batch in tqdm(val_loader, leave=False):
                        bpd, ok, _ = train_or_test(args, model, tokenizer, batch,
                                                   is_training=False, device=device)
                        if ok is not None:
                            acc_log.update(ok)
                        if bpd is not None:
                            bpd_log.update(bpd.item())

            if not is_required_training(args.method):
                break

            batch = next(iter(train_loader))
            train_or_test(args, model, tokenizer, batch, optimizer=opt,
                          is_training=True, device=device)

        save_checkpoint(model, config, args, k)

        save_probs(args, model, tokenizer, full_loader, device=device,
                   name=f"train_n_val_model{k}")
        save_probs(args, model, tokenizer, test_loader, device=device,
                   name=f"test_model{k}")

    average_ensemble_csvs(args.prob_output_folder, "train_n_val", args.num_ensembles)
    average_ensemble_csvs(args.prob_output_folder, "test", args.num_ensembles)