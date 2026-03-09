## 2026-03-08 Sunday
- Combined multi-head attention, feed-forward, residual connections, and layer norm.
- Residual connections help gradients flow.
- Layer norm stabilizes training.
- This is the building block of modern LLMs.

## 2026-03-07 Saturday
- Extended single-head to multi-head attention.
- Each head learns different relationships (syntax, semantics, coreference).
- Concatenated outputs and projected back to original dimension.
- This lets the model focus on multiple aspects of language simultaneously.

## 2026-03-06 Friday
- Built single-head self attetnion from scratch using NumPy.
- Q, K, V: Query looks for matches, Key offers info, Value carries it.
- Implemented softmax for numerical stability.
- Realized this is how models understand what "it" refers to in a sentence.
- This is the core of transformers which is the foundation of all LLMs.