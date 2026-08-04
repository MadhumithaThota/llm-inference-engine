import torch

def sample_next_token(
    logits,
    generated_tokens,
    repetition_penalty,
    temperature,
    top_k,
    top_p,
):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if repetition_penalty != 1.0:

        for token_id in set(generated_tokens):

            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    logits = logits / temperature

    # -----------------------
    # Top-k
    # -----------------------
    if top_k > 0:
        values, _ = torch.topk(logits, k=top_k)

        min_topk = values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < min_topk,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # -----------------------
    # Top-p
    # -----------------------

    if top_p < 1.0:

        # Sort logits
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        # Convert to probabilities
        sorted_probs = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        # Running cumulative probability
        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        # Remove tokens after top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original vocabulary order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        logits = logits.masked_fill(
            indices_to_remove,
            float("-inf"),
        )

    probs = torch.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probs,
        num_samples=1,
    )
