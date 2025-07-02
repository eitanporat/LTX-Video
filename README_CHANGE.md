# Question and Answers
Question: Is there anything else in Attention that needs to be updated to
properly support padded and packed inputs?

Answer: 
Inside of attention.py
RoPE should be adapted to worked with different chunks.

Outside of attention.py
- The loss needs to be masked in the mask tokens.
- Data Loaders for training need to be updated

# Packing Strategy Discussion
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

I read online that it is beneficial to also sort by length.
As a preprocessing step, I would sort the sequences by length.

# Design changes
- I modified the `prepare_attention_mask`
it creates a boolean mask based on
`mask_bool = ~(
    (q_segment_ids.unsqueeze(-1) != kv_segment_ids.unsqueeze(-2)) |
    (q_pad.unsqueeze(-1) | kv_pad.unsqueeze(-2))
)`

it matches kv segments to q segments and checks that the attention is not computed on padding tokens

additionally because I don't want nan's

I added
```
if hidden_states_segment_ids is not None:
    pad_rows = (hidden_states_segment_ids == 0).unsqueeze(-1) 
    hidden_states_a = hidden_states_a.masked_fill(pad_rows, 0.0)
```
I set zeros where the hidden_states_segment_ids are zero (mask tokens)

I also made it work with flash attention and sdpa attention.

for Flash attention we just pass `hidden_states_segment_ids` and `encoder_hidden_states_segment_ids` directly as arguments

```hidden_states_a = flash_attention(
    q=query,
    k=key,
    v=value,
    q_segment_ids=hidden_states_segment_ids,
    kv_segment_ids=encoder_hidden_states_segment_ids,
    sm_scale=attn.scale,
)```

I also added some backwards compatibility
```
if attention_mask is not None:
    attention_mask = attn.prepare_attention_mask_old(
        attention_mask, sequence_length, batch_size
    )
    # scaled_dot_product_attention expects attention_mask shape to be
    # (batch, heads, source_length, target_length)
    attention_mask = attention_mask.view(
        batch_size, attn.heads, -1, attention_mask.shape[-1]
    )
else:
    attention_mask = attn.prepare_attention_mask(
        kv_segment_ids=encoder_hidden_states_segment_ids, q_segment_ids=hidden_states_segment_ids, target_length=sequence_length, batch_size=batch_size, out_dim=4,
```

# Assumptions
I assumed that when calling attention either attention_mask is passed and then we revert to previous functionality or encoder_hidden_states_segment_ids and hidden_states_segment_ids are passed or all all arguments are None.

I verify that if `encoder_hidden_states_segment_ids` is not None then also `hidden_states_segment_ids` is not None.

# Tests I ran
I created a _make_attention attention function

Test 1: We compare the computed mask to a manually chosen mask.
Test 2: Check that the mask computation broadcasts correctly over the batch axis.
Test 3:  This test checks that the output of running attention on a padded sequence that contains two logical
    segments is the same as running attention separately for each segment.
Test 4:  This test checks that the output of running attention with no segment IDs is the same as running attention with all ones as segment IDs.
Test 5:  Test that the attention mask (float) and segment ids give the same result when there is just one real segment (ID == 1).
Test 6: I ran the inference.py script and verified I got the same outcome.

# Performance Implications
By adding more segments in a batch we can read the memory of the model once.
Also, we have one kernel call instead of multiple kernel calls.

We also use more more of the theoretical FLOPs of the GPU by increasing the number of computations we can ensure that all threads are working.

Flash attention supports block-diagonal attention. So it would also be faster and gain from this attention masking.

# Where I used AI
I used AI to write the tests, I gave an explicit description of what I wanted the tests to check and it created it.

I used it for researching STG.

Occasional bugs, for example there is a common bug in headless linux distributions
`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

I fixed it with:
```sudo apt update
sudo apt install -y libgl1```

# TODOs
- Check TPU Flash attention, I didn't check that this works.
- I ran my tests on a rented H100.
- Add compatibility for the inference.py script
- Modify the model, data loaders, positional embedding to support segmented attention.