from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculate BLEU score for a single reference and hypothesis.
    
    Args:
        reference: Reference text (string)
        hypothesis: Generated text to evaluate (string)
        weights: Weights for different n-grams (default: equal weights for 1-4 grams)
        
    Returns:
        BLEU score (between 0 and 1)
    """
    # Initialize smoothing function for BLEU score
    smoothie = SmoothingFunction().method1
    
    # Tokenize inputs
    tokenized_ref = nltk.word_tokenize(reference.lower())
    tokenized_hyp = nltk.word_tokenize(hypothesis.lower())
    
    # Calculate BLEU score
    return sentence_bleu([tokenized_ref], tokenized_hyp, weights=weights, smoothing_function=smoothie)

def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores between a single reference and hypothesis text.
    
    Args:
        reference: Reference text (string)
        hypothesis: Generated text to evaluate (string)
        
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    rouge = Rouge()
    
    # Handle empty text
    if not hypothesis.strip():
        return {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }
    
    try:
        # The Rouge library automatically calculates all three metrics:
        # - rouge-1: Unigram overlap (1-word matches)
        # - rouge-2: Bigram overlap (2-word sequences)
        # - rouge-l: Longest Common Subsequence between texts
        # Each metric includes precision (p), recall (r), and f1-score (f)
        return rouge.get_scores(hypothesis, reference)[0]
    except Exception as e:
        print(f"Error calculating ROUGE: {e}")
        return {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }

def test_metrics(reference, hypothesis):
    """
    Test BLEU and ROUGE scores.
    
    Args:
        reference (str): The reference text
        hypothesis (str): The generated text to evaluate
    """
    # BLEU score calculation
    bleu_score = calculate_bleu(reference, hypothesis)
    print(f"Score BLEU: {bleu_score:.4f}")
    
    # ROUGE score calculation
    rouge_scores = calculate_rouge(reference, hypothesis)
    
    # Print ROUGE F1 scores : harmonic mean of precision and recall
    print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")

if __name__ == "__main__":
    reference = "Climate change remains one of the biggest challenges facing humanity in the history ."
    hypothesis = "Climate change remains one of the most common events in the United States ."
    
    print("Référence:", reference)
    print("Hypothèse:", hypothesis)
    test_metrics(reference, hypothesis)