"""Evaluate the models on the MIR task of cross-modal retrieval"""

import torch
from torch.utils.data import DataLoader
from transformers import ClapProcessor, ClapModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torch.nn.functional as F
import argparse
import os
import sys

# Make the repo root importable (scripts/ lives one level below it).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the DALIDataset
from Dataset.DALIDataset import DALIDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)


def load_model(model_path):
    """Load the model
    
    Args:
        model_path: str, the path to the model
    
    Returns:
        model: ClapModel, the model"""
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        model = ClapModel.from_pretrained(model_path)
    return model

def collate_fn(batch):
    text, audio_data = zip(*batch)
    waveforms, sample_rates = zip(*audio_data)
    max_len = max(w.shape[1] for w in waveforms)
    padded_waveforms = torch.stack([torch.nn.functional.pad(w, (0, max_len - w.shape[1])) for w in waveforms])
    return text, padded_waveforms, torch.tensor(sample_rates)

def cross_modal_retrieval(model, processor, test_loader):
    """Evaluate the model on the cross-modal retrieval task
    
    Args:
        test_loader: DataLoader, the data loader for the test set
    
    Returns:
        top_k_accs: dict, the top k accuracy for each k
        kl_divs: float, the KL divergence between the audio and text distributions"""

    model.eval()
    loader = tqdm(test_loader)
    batch_size = 8

    ks = [2 ** i for i in range(batch_size) if 2 ** i < batch_size]
    top_k_accs = {k: [] for k in ks}
    kl_divs = []

    for batch in loader:
        lyrics, audio, sample_rates = batch
        audio = list(audio.squeeze().numpy())
        inputs = processor(text=lyrics, audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # get the classifications of the audio across the text
        audio_distribution = F.log_softmax(outputs.logits_per_audio, dim=-1)

        # get the top k accuracy
        for k in ks:
            top_k = torch.topk(audio_distribution, k, dim=-1).indices
            top_k_acc = (top_k == torch.arange(batch_size).unsqueeze(-1)).any(dim=-1).sum().item() / batch_size
            top_k_accs[k].append(top_k_acc)

        # text-to-text similarity: each lyric/description vs. every other one.
        # (Previously this compared each vector with itself, which is always 1.0
        # and makes the KL term meaningless.)
        text_embeds = F.normalize(outputs.text_embeds, dim=-1)
        text_distribution = F.log_softmax(text_embeds @ text_embeds.t(), dim=-1)

        # KL of the audio->text distribution from the text->text distribution
        kl_div = F.kl_div(audio_distribution, text_distribution, reduction="batchmean", log_target=True)
        kl_divs.append(kl_div.item())

        # set the progress bar description
        loader.set_description(f"KL Div: {np.mean(kl_divs):.2f}, Top 1 Acc: {np.mean(top_k_accs[1]):.2f}")

    return ({k: float(np.mean(accs)) for k, accs in top_k_accs.items()},
            float(np.mean(kl_divs)))


if __name__ == "__main__":
    # get CLI arguments for the model path
    parser = argparse.ArgumentParser(description="Evaluate the model on the cross-modal retrieval task")
    parser.add_argument("--model_path", type=str, default="laion/larger_clap_general", help="The path to the model to evaluate. For fine-tuned models, this should be a local path. Otherwise, the pretrained CLAP model is loaded.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for the data loader")
    parser.add_argument("--results_dir", type=str, default="./results/cross-modal-results/", help=r"The folder to save the results to. The results will be saved as a JSON file with the top-k accuracies and KL divergence. The file name is given as {model_path}-{batch_size}.json.")
    parser.add_argument("--use_sentiment", action="store_true", help="Evaluate the sentiment-description arm instead of the raw lyric.")

    args = parser.parse_args()
    batch_size = args.batch_size
    model_path = args.model_path
    results_path = args.results_dir

    # print run configuration
    print(f"Model Path: {model_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Results Directory: {results_path}")

    # Load the model
    model = load_model(model_path)

    # Load the processor
    processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

    # Load the dataset

    dataset = DALIDataset(use_sentiment=args.use_sentiment)
    dataset_len = len(dataset) #[0.01, 0.99]
    _, _, test_set = torch.utils.data.random_split(dataset, [8, 16, dataset_len - 24], generator=torch.Generator().manual_seed(SEED))
    # drop_last keeps every batch full so the diagonal ground-truth (arange over
    # batch_size) and the top-k logic stay valid on the final batch.
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    # Evaluate the model
    top_k_accs, kl_divs = cross_modal_retrieval(model, processor, test_loader)

    # plot the top-k accuracy
    plt.plot(list(top_k_accs.keys()), list(top_k_accs.values()))
    plt.xlabel("k")
    plt.ylabel("Top-k Accuracy")
    plt.title("Top-k Accuracy")
    plt.show()

    # print the KL divergence
    print(f"KL Divergence: {kl_divs:.2f}")
    print(f"Top-k Accuracies: {top_k_accs}")

    # create the results directory if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path) 
    
    model_path = model_path.split("/")[-1]
    output_path = os.path.join(results_path, f"{model_path}-{batch_size}.json")

    # save the results
    with open(output_path, "w") as f:
        json.dump({"top_k_accs": top_k_accs, "kl_divs": kl_divs}, f)

    print(f"Results saved to {output_path}")
