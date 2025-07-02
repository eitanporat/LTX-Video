# Question and Answers

**Question:**

Is there anything else in Attention that needs to be updated to properly support padded and packed inputs?

**Answer:**

### Inside `attention.py`

- RoPE should be adapted to work with different chunks.

### Outside `attention.py`

- The loss needs to be masked in the masked tokens.
- Data loaders for training need to be updated.

---

# Packing Strategy Discussion

- The most efficient way to pack sequences in general is a hard problem.
- One could take a greedy approach where we define a maximum `seqlen` for the concatenated sequences.
- We add sequences and also pad them so they all share the maximum sequence length.

### Pseudocode

```python
# GIVEN: MAX_SEQLEN, LEN = [L[1], ..., L[n]]

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
```

> 💡 *I read online that it is beneficial to also sort by length. As a preprocessing step, I would sort the sequences by length.*

---

# Design Changes

I modified the `prepare_attention_mask`. It creates a boolean mask based on:

```python
mask_bool = ~(
    (q_segment_ids.unsqueeze(-1) != kv_segment_ids.unsqueeze(-2)) |
    (q_pad.unsqueeze(-1) | kv_pad.unsqueeze(-2))
)
```

It matches `kv` segments to `q` segments and checks that the attention is not computed on padding tokens.

Additionally, to avoid `NaN`s, I added:

```python
if hidden_states_segment_ids is not None:
    pad_rows = (hidden_states_segment_ids == 0).unsqueeze(-1)
    hidden_states_a = hidden_states_a.masked_fill(pad_rows, 0.0)
```

This sets values to zero where `hidden_states_segment_ids` are zero (masked tokens).

---

### Flash Attention Integration

For Flash Attention, we pass `hidden_states_segment_ids` and `encoder_hidden_states_segment_ids` directly:

```python
hidden_states_a = flash_attention(
    q=query,
    k=key,
    v=value,
    q_segment_ids=hidden_states_segment_ids,
    kv_segment_ids=encoder_hidden_states_segment_ids,
    sm_scale=attn.scale,
)
```

---

### Backwards Compatibility

```python
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
        kv_segment_ids=encoder_hidden_states_segment_ids,
        q_segment_ids=hidden_states_segment_ids,
        target_length=sequence_length,
        batch_size=batch_size,
        out_dim=4,
    )
```

---

# Assumptions

- When calling attention:

  - Either `attention_mask` is passed → revert to previous functionality.
  - Or both `encoder_hidden_states_segment_ids` and `hidden_states_segment_ids` are passed.
  - Or all attention-related arguments are `None`.

- I verify that if `encoder_hidden_states_segment_ids` is not `None`, then `hidden_states_segment_ids` must also be not `None`.

---

# Tests I Ran

I created a `_make_attention` function.

**Test 1:** Compare the computed mask to a manually chosen mask.\
**Test 2:** Verify mask computation broadcasts over the batch axis.\
**Test 3:** Run attention on a padded sequence with two logical segments and compare to running attention separately per segment.\
**Test 4:** Verify output is identical when using all-ones segment IDs versus no segment IDs.\
**Test 5:** Confirm that `attention_mask` and segment IDs give the same result when there's one real segment (ID = 1).\
**Test 6:** Run the `inference.py` script and verify that the output remains the same.

---

# Performance Implications

- By adding more segments in a batch, we read the model’s memory once.
- We have one kernel call instead of many, improving efficiency.
- We utilize more of the GPU's theoretical FLOPs, maximizing thread utilization.
- Flash Attention supports block-diagonal attention, which makes it even faster with this masking approach.

---

# Where I Used AI

- I used AI to write the tests. I described the expected behavior and received implementations.
- I used AI for researching STG.
- AI helped debug:
  ```text
  ImportError: libGL.so.1: cannot open shared object file: No such file or directory
  ```
  Fixed with:
  ```bash
  sudo apt update
  sudo apt install -y libgl1
  ```
- Formatting this readme 😃
---

# TODOs

- ✅ Check TPU Flash Attention (currently **not tested**).
- ✅ Ran tests on a rented H100 GPU.
- 💪 Add full compatibility with the `inference.py` script.
- 💪 Modify the model, data loaders, and positional embeddings to support segmented attention.

