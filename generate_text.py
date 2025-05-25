import torch
import argparse
import os
from model import LongShortTermMemory
from data_processing import CorpusWords
from decoding_strategies import GreedyDecoding, BeamSearch, TopKSampling, NucleusSampling


def predict_next_words(model, dataset, seed_text, num_words=5, 
                      device='cpu', decoding_strategy=None, top_k=10, top_p=0.9):
    """
    Predict the next words given a seed text using the trained model.
    """
    model.eval()
    
    # Determine which decoding strategy to use
    if decoding_strategy is None or decoding_strategy == 'greedy':
        decoder = GreedyDecoding()
    elif decoding_strategy == 'topk':
        decoder = TopKSampling(k=top_k)
    elif decoding_strategy == 'nucleus':
        decoder = NucleusSampling(p=top_p)
    
    # Converts the seed text to tokens
    words = seed_text.strip().split()
    tokens = [dataset.dictionary.word2idx.get(w, dataset.dictionary.word2idx['<UNK>']) 
             for w in words]
    
    input_tensor = torch.LongTensor([tokens]).to(device)
    
    generated = tokens.copy()
    hidden = None
    
    with torch.no_grad():
        for _ in range(num_words):
            current_input = input_tensor
            output, hidden = model(current_input, hidden)
            # Probabilities of the next word
            word_weights = output[0, -1, :].exp()
            # Use the specified decoding strategy
            word_idx = decoder(word_weights)
            # Stop if we reach the end of the sequence ie <eos>
            if word_idx == dataset.dictionary.word2idx['<EOS>']:
                break
            generated.append(word_idx)
            # Update the input tensor with the new word
            input_tensor = torch.LongTensor([[word_idx]]).to(device)

    return dataset.decode(torch.tensor(generated))


def predict_with_beam_search(model, dataset, seed_text, num_words=5, beam_width=5, device='cpu'):
    """
    Predict the next words using beam search decoding.
    """
    model.eval()
    
    # Convert seed text to tokens
    words = seed_text.strip().split()
    tokens = [dataset.dictionary.word2idx.get(w, dataset.dictionary.word2idx['<UNK>']) 
             for w in words]
    
    input_tensor = torch.LongTensor([tokens]).to(device)
    hidden = None
    
    # Initialize beam search
    beams = [(tokens.copy(), 0.0)]  # (tokens, score)
    finished_beams = []
    eos_token = dataset.dictionary.word2idx['<EOS>']
    
    # Create beam search decoder
    beam_search = BeamSearch(beam_width=beam_width)
    
    # Generate words
    with torch.no_grad():
        for _ in range(num_words):
            # Process each beam
            all_candidates = []
            
            for beam_tokens, beam_score in beams:
                # Skip if this beam is already finished
                if beam_tokens[-1] == eos_token:
                    finished_beams.append((beam_tokens, beam_score))
                    continue
                
                # Prepare input for this beam
                beam_input = torch.LongTensor([beam_tokens]).to(device)
                
                # Run model for single step
                output, hidden_beam = model(beam_input, hidden)
                
                # Get logits for the next token
                next_token_logits = output[0, -1, :]
                
                # Get top-k tokens and their scores
                topk_logits, topk_indices = torch.topk(next_token_logits, beam_width)
                
                # Create candidates from this beam
                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    token_score = beam_score + topk_logits[i].item()
                    
                    # Create new candidate beam
                    new_beam_tokens = beam_tokens.copy() + [token_id]
                    all_candidates.append((new_beam_tokens, token_score))
                    
                    # Add to finished if EOS
                    if token_id == eos_token:
                        finished_beams.append((new_beam_tokens, token_score))
            
            # If we have enough finished beams, we're done
            if len(finished_beams) >= beam_width:
                break
                
            # Sort candidates by score and keep top-k
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
            
            # If all beams are finished, we're done
            if all(beam[0][-1] == eos_token for beam in beams):
                break
    
    # Combine finished beams with remaining beams
    all_beams = finished_beams + beams
    all_beams.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top beam's decoded text
    best_beam = all_beams[0][0]
    return dataset.decode(torch.tensor(best_beam))





def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset vocabulary...")
    # Just load the corpus without processing batches
    dataset = CorpusWords(args.data_path)
    vocab_size = dataset.get_vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Loading model ({args.model_type})...")
    model = LongShortTermMemory(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (validation loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
    
    print("\nGenerating text...")
    if args.decoding_strategy == 'greedy':
        generated_text = predict_next_words(
            model=model,
            dataset=dataset,
            seed_text=args.seed_text,
            num_words=args.num_words,
            device=device,
            decoding_strategy='greedy'
        )
    elif args.decoding_strategy == 'beam':
        generated_text = predict_with_beam_search(
            model=model,
            dataset=dataset,
            seed_text=args.seed_text,
            num_words=args.num_words,
            beam_width=args.beam_width,
            device=device
        )
    elif args.decoding_strategy == 'topk':
        generated_text = predict_next_words(
            model=model,
            dataset=dataset,
            seed_text=args.seed_text,
            num_words=args.num_words,
            device=device,
            decoding_strategy='topk',
            top_k=args.top_k
        )
    elif args.decoding_strategy == 'nucleus':
        generated_text = predict_next_words(
            model=model,
            dataset=dataset,
            seed_text=args.seed_text,
            num_words=args.num_words,
            device=device,
            decoding_strategy='nucleus',
            top_p=args.top_p
        )
    
    print("\nGenerated text:")
    print("-"*80)
    print(generated_text)
    print("-"*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a pre-trained language model")
    
    # Required arguments
    parser.add_argument("seed_text", type=str, 
                      help="Starting text to generate from")
    parser.add_argument("--num_words", type=int, default=10,
                      help="Number of words to generate")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="models/LongShortTermMemory_best.pt",
                      help="Path to the trained model")
    parser.add_argument("--model_type", type=str, choices=['lstm'], default='lstm',
                      help="Type of model to use")
    parser.add_argument("--data_path", type=str, default="wikitext2",
                      help="Path to dataset (for vocabulary)")
    
    # Model parameters (should match training parameters)
    parser.add_argument("--embedding_dim", type=int, default=300,
                      help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512,
                      help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                      help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                      help="Dropout rate")
    

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for generation (lower = more deterministic)")
    parser.add_argument("--decoding_strategy", type=str, default="greedy", 
                      choices=['greedy', 'beam', 'topk', 'nucleus'],
                      help="Strategy for decoding (greedy, beam search, top-k, or nucleus sampling)")
    parser.add_argument("--beam_width", type=int, default=5,
                      help="Beam width for beam search decoding")
    parser.add_argument("--top_k", type=int, default=10,
                      help="K value for top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.4,
                      help="P value for nucleus sampling (0.0-1.0)")
    parser.add_argument("--no_cuda", action="store_true",
                      help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    main(args)