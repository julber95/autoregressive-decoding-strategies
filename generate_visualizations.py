import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from model import LongShortTermMemory
from data_processing import CorpusWords

from plot_utils import run_all_visualizations

PROMPT = "The President of the United States"

def main():
    print(f"Generation of visualizations for the prompt: '{PROMPT}'")
    
    os.makedirs('utils', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        dataset_path = "wikitext2"
        dataset = CorpusWords(dataset_path)
        vocab_size = dataset.get_vocabulary_size()
        print(f"Dataset loaded - Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    try:
        model_path = 'models/language_model_best.pt'
        checkpoint = torch.load(model_path, map_location=device)
        
        model = LongShortTermMemory(
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=512,
            num_layers=2,
            dropout=0.5
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded {model_path}")
        print(f"Loss validation: {checkpoint['val_loss']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    run_all_visualizations(model, dataset, PROMPT, device)
    
    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()
