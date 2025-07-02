Question: Is there anything else in Attention that needs to be updated to
properly support padded and packed inputs?

Answer: 
Inside of attention.py
RoPE should be adapted to worked with different chunks.

Outside of attention.py
- The loss needs to be masked in the mask tokens.
- Data Loaders for training need to be updated

Packing Strategy Discussion
- The most efficient way to pack sequences in general is a hard problem
- One could take a greedy approach where we define a maximum seqlen for the concatenated sequences
- We add sequences and also pad sequences so they are all share the maxiumum sequence length.
- In pseudocode
GIVEN: MAX_SEQLEN, SEQUENCES = [L[1], ..., L[n]]

block_indices   = []          
block_max_len   = 0           
block_tokens    = 0          

for i, cur_len in enumerate(LEN):
    candidate_max = max(block_max_len, cur_len)
    candidate_tok = candidate_max * (len(block_indices) + 1)

    if candidate_tok > MAX_SEQLEN and block_indices:
        padding = [block_max_len - LEN[j] for j in block_indices]
        yield block_indices, [LEN[j] for j in block_indices], padding

        block_indices = []
        block_max_len = 0
        block_tokens  = 0

    block_indices.append(i)
    block_max_len = max(block_max_len, cur_len)
    block_tokens  = block_max_len * len(block_indices)

if block_indices:
    padding = [block_max_len - LEN[j] for j in block_indices]
    yield block_indices, [LEN[j] for j in block_indices], padding