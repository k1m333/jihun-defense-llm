import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    """Layer normalization (simplified, no learned parameters yet)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def multi_head_attention(X, num_heads=4):
    n, d = X.shape
    d_k = d // num_heads

    np.random.seed(42)
    W_Q = np.random.randn(num_heads, d, d_k)
    W_K = np.random.randn(num_heads, d, d_k)
    W_V = np.random.randn(num_heads, d, d_k)
    W_O = np.random.randn(num_heads * d_k, d)

    head_outputs = []
    for h in range(num_heads):
        Q = X @ W_Q[h]
        K = X @ W_K[h]
        V = X @ W_V[h]

        scores = Q @ K.T / np.sqrt(d_k)
        attn_weights = softmax(scores, axis=-1)
        head_out = attn_weights @ V
        head_outputs.append(head_out)

    concat = np.concatenate(head_outputs, axis=-1)
    attn_output = concat @ W_O
    return attn_output

def feed_forward(x, d_ff=32):
    """Simple FFN: linear -> ReLU -> linear."""
    np.random.seed(42)
    W1 = np.random.randn(x.shape[-1], d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, x.shape[-1])
    b2 = np.random.randn(x.shape[-1])

    hidden = np.maximum(0, x @ W1 + b1)  # ReLU
    return hidden @ W2 + b2

def transformer_block(X, num_heads=4, d_ff=32):
    """
    One transformer encoder block:
    1. Multi-head attention
    2. Residual connection + layer norm
    3. Feed-forward
    4. Residual connection + layer norm
    """
    # Multi-head attention + residual + norm
    attn_out = multi_head_attention(X, num_heads)
    X = layer_norm(X + attn_out)

    # Feed-forward + residual + norm
    ff_out = feed_forward(X, d_ff)
    X = layer_norm(X + ff_out)

    return X

if __name__ == "__main__":
    n, d = 3, 8
    X = np.random.randn(n, d)
    out = transformer_block(X, num_heads=4, d_ff=32)
    print("Output shape:", out.shape)  # Should be (3, 8)
