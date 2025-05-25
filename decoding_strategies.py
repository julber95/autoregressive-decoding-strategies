import torch
import torch.nn.functional as F
import numpy as np

class GreedyDecoding:
    """
    Implements greedy decoding strategy: always select the token with highest probability.
    """
    def __call__(self, word_weights):
        """
        Select the next token using greedy search (argmax).
        
        Args:
            word_weights: Tensor containing token probabilities
            
        Returns:
            Index of the token with highest probability
        """
        return torch.argmax(word_weights).item()
    

class TopKSampling:
    """
    Implements top-k sampling: randomly samples from the k most likely tokens.
    """
    def __init__(self, k=10):
        self.k = k
        
    def __call__(self, word_weights):
        """
        Select the next token using top-k sampling.
        
        Args:
            word_weights: Tensor containing token probabilities
            
        Returns:
            Index of the selected token
        """
        # Get the top k tokens
        top_k_weights, top_k_indices = torch.topk(word_weights, min(self.k, len(word_weights)))
        
        # Normalize the probabilities of just these tokens
        top_k_weights = top_k_weights / top_k_weights.sum()
        
        # Sample from the distribution
        selected_idx = torch.multinomial(top_k_weights, 1).item()
        
        # Get the actual token index
        return top_k_indices[selected_idx].item()
    

class NucleusSampling:
    """
    Implements nucleus sampling.
    Selects from the smallest possible set of tokens whose cumulative probability exceeds p.
    """
    def __init__(self, p=0.9):
        self.p = p
        
    def __call__(self, word_weights):
        """
        Select the next token using nucleus sampling.
        
        Args:
            word_weights: Tensor containing token probabilities
            
        Returns:
            Index of the selected token
        """
        # Sort the probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(word_weights, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask for tokens in the nucleus (cumulative prob < p)
        nucleus_mask = cumulative_probs < self.p
        
        # Add the first token that exceeds p to be included in the nucleus
        nucleus_mask[0] = True  # Always include at least the most likely token
        
        # Find the last index where cumulative_prob < p
        last_included = torch.where(nucleus_mask)[0][-1].item()
        
        # Get the indices and probabilities in the nucleus
        nucleus_indices = sorted_indices[:last_included + 1]
        nucleus_probs = sorted_probs[:last_included + 1]
        
        # Normalize the probabilities
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        
        # Sample from the nucleus distribution
        selected_idx = torch.multinomial(nucleus_probs, 1).item()
        
        # Return the actual token index
        return nucleus_indices[selected_idx].item()



class BeamSearch:
    """
    Implements beam search decoding: maintains multiple hypothesis paths.
    """
    def __init__(self, beam_width=5):
        self.beam_width = beam_width
    
    def generate(self, model, input_ids, max_length=50, device='cpu'):
        """
        Generate a sequence using beam search.
        
        Args:
            model: The language model
            input_ids: Initial token IDs
            max_length: Maximum generation length
            device: Device to run on
            
        Returns:
            Tensor of generated tokens
        """
        model.eval()
        
        # Convert input_ids to tensor if it's not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(device)
        
        # Initialize beams with the input sequence
        beams = [(input_ids[0].tolist(), 0.0)]
        finished_beams = []
        
        # Generate up to max_length tokens
        with torch.no_grad():
            for _ in range(max_length):
                all_candidates = []
                
                # Process each beam
                for beam_tokens, beam_score in beams:
                    # Skip finished beams
                    if beam_tokens[-1] == model.eos_token_id if hasattr(model, 'eos_token_id') else False:
                        finished_beams.append((beam_tokens, beam_score))
                        continue
                    
                    # Prepare input for this beam
                    beam_input = torch.LongTensor([beam_tokens]).to(device)
                    
                    # Get model output
                    output, _ = model(beam_input)
                    logits = output[0, -1, :]
                    next_token_probs = F.softmax(logits, dim=0)
                    
                    # Get top-k tokens and probabilities
                    topk_probs, topk_indices = torch.topk(
                        next_token_probs, 
                        self.beam_width
                    )
                    
                    # Add candidates from this beam
                    for i in range(self.beam_width):
                        token_id = topk_indices[i].item()
                        token_prob = topk_probs[i].item()
                        
                        # Log probabilities for numerical stability
                        new_score = beam_score + np.log(token_prob)
                        new_tokens = beam_tokens + [token_id]
                        
                        all_candidates.append((new_tokens, new_score))
                
                # If we have enough finished beams or no candidates, stop
                if len(finished_beams) >= self.beam_width or not all_candidates:
                    break
                
                # Sort candidates by score (higher is better)
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the top-k beams
                beams = all_candidates[:self.beam_width]
        
        # Combine finished beams with current beams
        all_beams = finished_beams + beams
        all_beams.sort(key=lambda x: x[1], reverse=True)
        
        # Return the tokens of the best beam
        return torch.tensor(all_beams[0][0] if all_beams else input_ids[0])