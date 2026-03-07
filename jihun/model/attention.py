import numpy as np

def softmax(x, axis=1):
    """Stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """
    Single-head self-attention.
    X : (n, d) - input embeddings
    W_Q, W_K, W_V : (d, d_k) - learned weights (d_k = d for simplicity)
    Returns:
        output: (n, d_k) - contest-aware representations
        attention_weights: (n, n) - for inspection
    """
    Q = X @ W_Q # (n, d_k)
    K = X @ W_K # (n, d_k)
    V = X @ W_V # (n, d_k)

    d_k = W_Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k) # (n, n)
    attention_weights = softmax(scores, axis=1)
    output = attention_weights @ V # (n, d_k)

    return output, attention_weights

if __name__ == "__main__":
    np.random.seed(42)
    n, d = 3, 4
    X = np.random.randn(n, d)
    W_Q = np.random.randn(d, d)
    W_K = np.random.randn(d, d)
    W_V = np.random.randn(d, d)

    out, attn = self_attention(X, W_Q, W_K, W_V)
    print("Output shape:", out.shape)
    print("Attention weights:\n", attn)
