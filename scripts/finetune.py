import torch
from torch.utils.data import DataLoader
from torch import optim
from transformers import ClapProcessor, ClapModel
from Dataset.DALIDataset import DALIDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_FOLDER = "./model"
SEED = 42
torch.manual_seed(SEED)

# NOTE: Try other model configurations
model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

def collate_fn(batch):
    text, audio_data = zip(*batch)
    waveforms, sample_rates = zip(*audio_data)
    max_len = max(w.shape[1] for w in waveforms)
    padded_waveforms = torch.stack([torch.nn.functional.pad(w, (0, max_len - w.shape[1])) for w in waveforms])
    return text, padded_waveforms, torch.tensor(sample_rates)

def validate(test_dl):
    model.eval()
    test_loss=0.0
    print("validating")
    tbar = tqdm(test_dl)
    for batch in tbar:
        lyrics, audio, sample_rates = batch
        audio = list(audio.squeeze().numpy())
        inputs = processor(text=lyrics, audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, return_loss=True)
        print("got outputs")
        test_loss += outputs.loss.item()
        print("test_loss", test_loss)
    print("done!")
    test_total_loss=test_loss/len(test_dl)
    print(f"Validation Loss {test_total_loss:.3f}")
    return test_total_loss

# Load the dataset
batch_size = 8 # paper had 768
dataset = DALIDataset(use_sentiment=False)
dataset_len = len(dataset) #[0.01, 0.99]
train_set, val_set, _ = torch.utils.data.random_split(dataset, [8, 16, dataset_len - 24], generator=torch.Generator().manual_seed(SEED))
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

num_epochs = 2 # paper had 45
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.99, 0.9), eps=1e-9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_dataloader))

pbar = tqdm(train_dataloader)

num_training_steps = num_epochs * len(train_dataloader)

torch.cuda.empty_cache()
print(f"Using {device} for training")
best_val_loss = 9999999
history = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
    train_loss=0.0
    model.train()
    for batch in pbar:
        optimizer.zero_grad()
        print("batch")
        lyrics, audio, sample_rates = batch

        # NOTE: not all formats work with ClapFeatureExtractor: https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/clap/feature_extraction_clap.py#L33
        audio = list(audio.squeeze().numpy())
        inputs = processor(text=lyrics, audios=audio, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, return_loss=True)

        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # NOTE: not sure if this is correct
        scheduler.step()
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

    history['train_loss'].append(train_loss/len(train_dataloader))
    val_loss = validate(val_dataloader)
    history['val_loss'].append(val_loss)
    if val_loss < best_val_loss:
        print("Better score reached, saving checkpoint...")
        if os.path.exists(os.path.join(MODEL_SAVE_FOLDER, "best_model.pt")):
            os.remove(os.path.join(MODEL_SAVE_FOLDER, "best_model.pt"))
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_FOLDER, "best_model.pt"))

        # save the model history
        with open(os.path.join(MODEL_SAVE_FOLDER, "history.json"), "w") as f:
            json.dump(history, f)

        f, ax = plt.subplots(figsize=(10,6))
        plt.plot(history['val_loss'],label="validation")
        plt.plot(history['train_loss'],label="train")
        plt.legend()
        plt.suptitle(f"Training & Validation Loss during Fine-tuning, Epoch{epoch}");
        plt.savefig(os.path.join(MODEL_SAVE_FOLDER, "loss.png"))
        plt.close()