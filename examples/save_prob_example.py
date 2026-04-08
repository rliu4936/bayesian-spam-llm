import os
import argparse
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from autograder.dataset import CPEN455_2025_W1_Dataset
from model import LlamaModel
from model.config import Config
from model.tokenizer import Tokenizer
from utils.device import set_device
from examples.bayes_inverse import save_probs, average_ensemble_csvs


# load one checkpoint
def load_checkpoint(ckpt_path, device):
    torch.serialization.add_safe_globals([Config])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    args = ckpt["args"]
    state_dict = ckpt["model_state"]

    model = LlamaModel(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model, config, args


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--test_dataset_path", type=str,
                        default="autograder/cpen455_released_datasets/test_subset.csv")
    parser.add_argument("--prob_output_folder", type=str, default="bayes_inverse_probs")
    parser.add_argument("--ensemble_size", type=int, default=9)
    args = parser.parse_args()

    load_dotenv()

    # device + tokenizer
    device = set_device()
    base_checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")
    tokenizer = Tokenizer.from_pretrained(base_checkpoint, cache_dir=model_cache_dir)

    # dataloader
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=args.test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.prob_output_folder, exist_ok=True)

    # ensemble loop
    models_saved = 0
    for k in range(args.ensemble_size):
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_ens{k}.pt")

        if not os.path.exists(ckpt_path):
            print(f"WARNING: missing checkpoint: {ckpt_path}")
            continue

        print(f"Loading {ckpt_path}")
        model, config, ckpt_args = load_checkpoint(ckpt_path, device)

        # wrap saved args into a simple object
        class TempArgs: pass
        a = TempArgs()
        for key, value in ckpt_args.items():
            setattr(a, key, value)

        save_probs(a, model, tokenizer, test_dataloader,
                   device=device, name=f"test_model{k}")

        print(f"Saved probabilities for model {k}")
        models_saved += 1

    # Average ensemble predictions
    if models_saved > 0:
        print(f"Averaging predictions from {models_saved} models...")
        average_ensemble_csvs(args.prob_output_folder, "test", models_saved)
        print(f"Saved ensemble predictions to {args.prob_output_folder}/test_dataset_probs.csv")
    else:
        print("WARNING: No models were found to generate predictions!")

    print("Done.")