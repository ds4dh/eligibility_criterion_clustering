import logging
import torch
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score
from dataclasses import dataclass
logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class BERTScoreOutput:
    precision: float
    recall: float
    fmeasure: float
    
    
def create_chunks(
    tokens: list[int],
    max_tokens: int=512,
    overlap: int=256,
) -> list[list[int]]:
    """ Split input token sequence into max_token lengths chunks, with overlap 
    
    Args:
        tokens (list[int]): input tokens
        max_tokens (int, optional): max number of tokens per chunk
        overlap (int, optional): amount of overlap between chunks, in tokens
        
    Returns:
        list[list[int]]: chunked but overlapping token sequence
    """
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunks.append(tokens[start_idx:end_idx])
        
        # Stop if we reached the end
        if end_idx == len(tokens):
            break
        
        # Move the start_idx forward by the size of max_tokens - overlap
        start_idx += max_tokens - overlap
    
    return chunks
    
    
def calculate_bertscore_with_sliding_window(
    target: str,
    prediction: str,
    scoring_model_id: str="allenai/scibert_scivocab_uncased",
    verbose: bool=False,
) -> BERTScoreOutput:
    """ Calculates BERTScore for long text sequences using a sliding window
    
    Args:
        target (str): reference text against which prediction is evaluated
        prediction (str): generated text that is evaluated
        scoring_model_id (str, optional): model used for scoring
        verbose (bool, optional): if True, detailed logging is provided
        
    Returns:
        BERTScoreOutput: An object containing the precision, recall, and F1 score averaged across all text chunks.
    """
    # Tokenize the texts
    tokenizer = AutoTokenizer.from_pretrained(scoring_model_id)
    target_tokens = tokenizer(target, return_tensors="pt")["input_ids"][0]
    predicted_tokens = tokenizer(prediction, return_tensors="pt")["input_ids"][0]
    
    # Create chunks for both reference and generated text
    target_chunks = create_chunks(target_tokens)
    predicted_chunks = create_chunks(predicted_tokens)
    
    # Adapt number of chunks in case they do not match
    if len(target_chunks) > 1:
        if verbose: print("Too long reference text will be chunked")
    if len(target_chunks) != len(predicted_chunks):
        if verbose: print("Length mismatch between reference and generated text chunks")
        if len(target_chunks) > len(predicted_chunks):
            if verbose: print("Extra chunks in reference text will be cut")
            target_chunks = target_chunks[:len(predicted_chunks)]
        else:
            if verbose: print("Extra chunks in generated text will be cut")
            predicted_chunks = predicted_chunks[:len(target_chunks)]
            
    # Decode chunks back to text
    target_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in target_chunks]
    predicted_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in predicted_chunks]
    
    # Checking sequence length with round-trip for SciBERT because it has a weird tokenizer
    if scoring_model_id == "allenai/scibert_scivocab_uncased":
        target_chunks = clean_chunks_for_scibert(target_chunks, tokenizer)
        predicted_chunks = clean_chunks_for_scibert(predicted_chunks, tokenizer)
        
    # Calculate BERTScore for all chunks at once
    precisions, recalls, fmeasures = score(
        cands=predicted_chunks,
        refs=target_chunks,
        model_type=scoring_model_id,
        verbose=verbose,
    )
    
    # Return scores averaged across chunks
    return BERTScoreOutput(
        precision=precisions.mean().item(),
        recall=recalls.mean().item(),
        fmeasure=fmeasures.mean().item(),
    )


def clean_chunks_for_scibert(
    chunks: list[str],
    tokenizer: AutoTokenizer,
) -> list[str]:
    """ Move excess tokens to the next chunk to avoid an error in SciBERT, since
        tokenizing a text of 512 tokens, decoding it and sending it back to the
        SciBERT tokenizer creates a 513 token sequence
    Args:
        chunks (list[str]): list of text chunks to be cleaned
    Returns:
        list[str]: cleaned list of text chunks
    """
    cleaned_chunks = []
    tokens_to_pass = torch.tensor([], dtype=torch.int64)
    for t in chunks:
        tokens = tokenizer(t, return_tensors="pt")["input_ids"][0]
        if len(tokens_to_pass) > 0:
            tokens = torch.cat((
                tokens[:1],  # [CLS]
                tokens_to_pass,  # from previous chunk
                tokens[1:]  # remaining tokens
            ))
        if len(tokens) > 512:
            tokens_to_pass = tokens[512 - 1:][:-1]  # excess tokens for this chunk    
        cleaned_tokens = torch.cat((tokens[:511], tokens[-1].unsqueeze(0)))  # [SEP]
        cleaned_chunk = tokenizer.decode(cleaned_tokens, skip_special_tokens=True)
        cleaned_chunks.append(cleaned_chunk)
    
    return cleaned_chunks


def compute_scores(
    reference: str,
    cluster_prediction: str,
    llm_prediction: str,
) -> dict:
    """ Compute diverse ROUGE-score-based metrics based on one sample of cluster
        prediction, corresponding LLM prediciton, and associated reference
    Args:
        reference (str): the reference text to compare against
        cluster_prediction (str): the cluster-generated prediction
        llm_prediction (str): the LLM-generated prediction
    """
    # Compute scores and average scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    cluster_scores = scorer.score(target=reference, prediction=cluster_prediction)
    llm_scores = scorer.score(target=reference, prediction=llm_prediction)
    cluster_avg_r = (cluster_scores["rouge1"].recall + cluster_scores["rouge2"].recall + cluster_scores["rougeL"].recall) / 3
    cluster_avg_p = (cluster_scores["rouge1"].precision + cluster_scores["rouge2"].precision + cluster_scores["rougeL"].precision) / 3
    cluster_avg_f = (cluster_scores["rouge1"].fmeasure + cluster_scores["rouge2"].fmeasure + cluster_scores["rougeL"].fmeasure) / 3
    llm_avg_r = (llm_scores["rouge1"].recall + llm_scores["rouge2"].recall + llm_scores["rougeL"].recall) / 3
    llm_avg_p = (llm_scores["rouge1"].precision + llm_scores["rouge2"].precision + llm_scores["rougeL"].precision) / 3
    llm_avg_f = (llm_scores["rouge1"].fmeasure + llm_scores["rouge2"].fmeasure + llm_scores["rougeL"].fmeasure) / 3
    
    cluster_scores.update({"BERT": calculate_bertscore_with_sliding_window(target=reference, prediction=cluster_prediction, scoring_model_id="bert-base-uncased")})
    cluster_scores.update({"SciBERT": calculate_bertscore_with_sliding_window(target=reference, prediction=cluster_prediction, scoring_model_id="allenai/scibert_scivocab_uncased")})
    cluster_scores.update({"Longformer": calculate_bertscore_with_sliding_window(target=reference, prediction=cluster_prediction, scoring_model_id="allenai/longformer-large-4096")})
    
    llm_scores.update({"BERT": calculate_bertscore_with_sliding_window(target=reference, prediction=llm_prediction, scoring_model_id="bert-base-uncased")})
    llm_scores.update({"SciBERT": calculate_bertscore_with_sliding_window(target=reference, prediction=llm_prediction, scoring_model_id="allenai/scibert_scivocab_uncased")})
    llm_scores.update({"Longformer": calculate_bertscore_with_sliding_window(target=reference, prediction=llm_prediction, scoring_model_id="allenai/longformer-large-4096")})
        
    # Prepare the result row for this clinical trial
    method_perfs_avg = {"Cluster": cluster_avg_f, "LLM": llm_avg_f}
    best_method_avg = max(method_perfs_avg, key=lambda k: method_perfs_avg[k])
    method_perfs_bert = {"Cluster": cluster_scores["BERT"].fmeasure, "LLM": llm_scores["BERT"].fmeasure}
    best_method_bert = max(method_perfs_bert, key=lambda k: method_perfs_bert[k])
    result_dict = {
        "Best Method Average": best_method_avg,
        "Best Method BERT": best_method_bert,
        
        
        "Cluster ROUGE-1-Recall Score": cluster_scores["rouge1"].recall,
        "Cluster ROUGE-1-Precision Score": cluster_scores["rouge1"].precision,
        "Cluster ROUGE-1-F Score": cluster_scores["rouge1"].fmeasure,
        
        "Cluster ROUGE-2-Recall Score": cluster_scores["rouge2"].recall,
        "Cluster ROUGE-2-Precision Score": cluster_scores["rouge2"].precision,
        "Cluster ROUGE-2-F Score": cluster_scores["rouge2"].fmeasure,
        
        "Cluster ROUGE-L-Recall Score": cluster_scores["rougeL"].recall,
        "Cluster ROUGE-L-Precision Score": cluster_scores["rougeL"].precision,
        "Cluster ROUGE-L-F Score": cluster_scores["rougeL"].fmeasure,
        
        "Cluster ROUGE-Average-R Score": cluster_avg_r,
        "Cluster ROUGE-Average-P Score": cluster_avg_p,
        "Cluster ROUGE-Average-F Score": cluster_avg_f,
        
        "Cluster BERT-Recall Score": cluster_scores["BERT"].recall,
        "Cluster BERT-Precision Score": cluster_scores["BERT"].precision,
        "Cluster BERT-F Score": cluster_scores["BERT"].fmeasure,
        
        "Cluster SciBERT-Recall Score": cluster_scores["SciBERT"].recall,
        "Cluster SciBERT-Precision Score": cluster_scores["SciBERT"].precision,
        "Cluster SciBERT-F Score": cluster_scores["SciBERT"].fmeasure,
        
        "Cluster Longformer-Recall Score": cluster_scores["Longformer"].recall,
        "Cluster Longformer-Precision Score": cluster_scores["Longformer"].precision,
        "Cluster Longformer-F Score": cluster_scores["Longformer"].fmeasure,
        
        
        "LLM ROUGE-1-Recall Score": llm_scores["rouge1"].recall,
        "LLM ROUGE-1-Precision Score": llm_scores["rouge1"].precision,
        "LLM ROUGE-1-F Score": llm_scores["rouge1"].fmeasure,
        
        "LLM ROUGE-2-Recall Score": llm_scores["rouge2"].recall,
        "LLM ROUGE-2-Precision Score": llm_scores["rouge2"].precision,
        "LLM ROUGE-2-F Score": llm_scores["rouge2"].fmeasure,
        
        "LLM ROUGE-L-Recall Score": llm_scores["rougeL"].recall,
        "LLM ROUGE-L-Precision Score": llm_scores["rougeL"].precision,
        "LLM ROUGE-L-F Score": llm_scores["rougeL"].fmeasure,
        
        "LLM ROUGE-Average-R Score": llm_avg_r,
        "LLM ROUGE-Average-P Score": llm_avg_p,
        "LLM ROUGE-Average-F Score": llm_avg_f,
        
        "LLM BERT-Recall Score": llm_scores["BERT"].recall,
        "LLM BERT-Precision Score": llm_scores["BERT"].precision,
        "LLM BERT-F Score": llm_scores["BERT"].fmeasure,
        
        "LLM SciBERT-Recall Score": llm_scores["SciBERT"].recall,
        "LLM SciBERT-Precision Score": llm_scores["SciBERT"].precision,
        "LLM SciBERT-F Score": llm_scores["SciBERT"].fmeasure,
        
        "LLM Longformer-Recall Score": llm_scores["Longformer"].recall,
        "LLM Longformer-Precision Score": llm_scores["Longformer"].precision,
        "LLM Longformer-F Score": llm_scores["Longformer"].fmeasure,
    }
    
    return result_dict