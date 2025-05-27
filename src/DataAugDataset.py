import torch
from torch.utils.data import Dataset
import random
import ast


class DataAugDataset(Dataset):
    def __init__(self, texts, keywords, paraphrases, labels, tokenizer, max_length, random_text=False, random_remove=False, random_order=False):
        super().__init__()
        self.texts = texts
        self.keywords = keywords
        self.paraphrases = paraphrases
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_text = random_text
        self.random_remove = random_remove
        self.random_order = random_order

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        aspect = self.keywords[idx]

        if self.random_text and random.random() <= self.random_text:
            paraphrase = self.paraphrases[idx]
            samples = ast.literal_eval(paraphrase)
            samples=list(samples)
            # Take one text randomly
            samples.append(text)
            text = random.choice(samples)
        
        words = text.split()
        
        # Randomly remove words
        if self.random_remove and random.random() < self.random_remove:
            if len(words) > 1:
                num_to_remove = max(1, int(0.1 * len(words)))  # Quitar ~10% de las palabras
                indices_to_remove = random.sample(range(len(words)), num_to_remove)
                words = [w for i, w in enumerate(words) if i not in indices_to_remove]
                text = " ".join(words)

        # Randomly shuffle words
        if self.random_order and random.random() < self.random_order:
            if len(words) > 1:
                random.shuffle(words)
                text = " ".join(words)

        # Tokenizar el texto principal
        encoding = self.tokenizer(text, aspect, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
