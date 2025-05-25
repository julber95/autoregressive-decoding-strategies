import torch
import torch.nn as nn
import os
from tqdm import tqdm
import argparse

from data_processing import main as process_data
from model import LongShortTermMemory

def train_model(model, train_inputs, train_targets, val_inputs, val_targets, 
              criterion, optimizer, num_epochs=10, device='cpu',
              save_path='models', model_name='LongShortTermMemory'):
    """
        Trains the language model with progress bar and checkpoint saving.
        
        Args:
            model: The model to train
            train_inputs, train_targets: Training data
            val_inputs, val_targets: Validation data
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of epochs
            device: Training device ('cpu' or 'cuda')
            save_path: Directory to save models
            model_name: Base name for saved files
        
        Returns:
            Tuple (train_losses, val_losses)
    """
    os.makedirs(save_path, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = train_inputs.size(0)
        
        train_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_bar:

            inputs = train_inputs[batch]   # [batch_size, seq_length]
            targets = train_targets[batch]  # [batch_size, seq_length]
            
            outputs, _ = model(inputs)  # [batch_size, seq_length, vocab_size]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) # To avoid exploding gradients
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}") 
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        num_val_batches = val_inputs.size(0)
    
        val_bar = tqdm(range(num_val_batches), desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for batch in val_bar:
                val_out, _ = model(val_inputs[batch])
                v_loss = criterion(val_out.reshape(-1, val_out.size(-1)), 
                                  val_targets[batch].reshape(-1))
                val_loss += v_loss.item()
                val_bar.set_postfix(loss=f"{v_loss.item():.4f}")
                
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_path, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"New best model saved: {best_model_path} (val_loss: {best_val_loss:.4f})")
    
    return train_losses, val_losses


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print("Load data...")
    dataset, train_inputs, train_targets, val_inputs, val_targets, _, _ = process_data(
        batch_size=args.batch_size, 
        seq_length=args.seq_length,
        device=device,
        verbose=args.verbose
    )
    
    vocab_size = dataset.get_vocabulary_size()
    print(f"Size of the vocabulary : {vocab_size}")

    print("Model initialization...")
    model = LongShortTermMemory(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    if args.verbose:
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters : {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model, 
        train_inputs=train_inputs, 
        train_targets=train_targets, 
        val_inputs=val_inputs, 
        val_targets=val_targets, 
        criterion=criterion, 
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_path='models',
        model_name="LSTM"
    )
    
    print("Training done.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for LSTM and Attention models")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=35)

    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden size of LSTM layers")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.5)
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=30)
    

    parser.add_argument("--model_name", type=str, default="LSTM",
                        help="Model name to save the checkpoint")
    
    parser.add_argument("--verbose", action="store_true",
                        help="To print the model summary and total number of parameters")
    
    args = parser.parse_args()
    
    main(args)