import torch            # PyTorch: the deep learning framework that handles tensors (multi-dim arrays),
                        # automatic differentiation (backprop), and GPU acceleration.

import torch.nn as nn   # nn = "neural network" module. Contains all the building blocks:
                        # layers (Linear, Embedding), loss functions, containers (Module, ModuleList).
                        # We alias it as "nn" so we can write nn.Linear instead of torch.nn.Linear.

import math             # Python's built-in math library. We only need math.sqrt() and math.log().


# =============================================================================
# COMPONENT 1: INPUT EMBEDDINGS
# =============================================================================
# PURPOSE: Convert token IDs (integers) into dense vectors the model can work with.
#
# ANALOGY: Think of it as a dictionary.
#   - You look up a word (token ID) → you get its definition (a vector of 512 numbers).
#   - But instead of a human-written definition, the "definition" is 512 numbers
#     that the model LEARNS during training to capture meaning.
#
# FOR OUR PROJECT:
#   C++ input:  "int x = 5 ;"  → tokenizer → [34, 102, 8, 55, 11]
#   Each ID → looked up in embedding table → 5 vectors of size 512
# =============================================================================


class InputEmbeddings(nn.Module):
    # nn.Module is the BASE CLASS for every neural network component in PyTorch.
    # By inheriting from it, our class gets:
    #   - Automatic parameter tracking (weights are registered for training)
    #   - .to(device) support (move to GPU/CPU)
    #   - .parameters() method (optimizer uses this to update weights)
    #   - Ability to be nested inside other modules

    def __init__(self, d_model: int, vocab_size: int):
        # __init__ = constructor, called when you do: InputEmbeddings(512, 10000)
        #
        # d_model:    dimension of each embedding vector.
        #             The original paper uses 512.
        #             Meaning: each token becomes a list of 512 numbers.
        #
        # vocab_size: total number of unique tokens in our vocabulary.
        #             For code translation, might be ~8000-15000 tokens.

        super().__init__()
        # REQUIRED CALL. This runs nn.Module's constructor which sets up:
        #   - Internal dict to track parameters (self._parameters)
        #   - Internal dict to track sub-modules (self._modules)
        #   - Training/eval mode flag
        # If you forget this line, PyTorch won't know about your layers.

        self.d_model = d_model
        # Save d_model so we can use it in forward().
        # We need it for the sqrt(d_model) scaling.

        self.vocab_size = vocab_size
        # Save vocab_size (not strictly needed in forward, but good practice).

        self.embedding = nn.Embedding(vocab_size, d_model)
        # THIS IS THE CORE.
        #
        # nn.Embedding creates a 2D matrix of shape (vocab_size x d_model).
        # Example: if vocab_size=10000 and d_model=512, this is a (10000 x 512) table.
        #
        #   Row 0:    [0.12, -0.45, 0.78, ..., 0.33]   <- 512 numbers for token 0
        #   Row 1:    [0.56, 0.23, -0.11, ..., -0.89]  <- 512 numbers for token 1
        #   Row 2:    [...]                              <- 512 numbers for token 2
        #   ...
        #   Row 9999: [...]                              <- 512 numbers for token 9999
        #
        # When you pass token ID 45, it simply RETURNS row 45.
        # These values start RANDOM and are updated during training.
        # Over time, tokens with similar meanings get similar vectors.

    def forward(self, x):
        # forward() defines what happens when you CALL the module: embeddings(x)
        # PyTorch calls forward() automatically — you never call it directly.
        #
        # x: a tensor of token IDs
        #    Shape: (batch_size, seq_len)
        #    Example: batch_size=2, seq_len=5
        #    x = [[34, 102, 8, 55, 11],    <- C++ program 1 (tokenized)
        #         [22, 67, 90, 14, 38]]     <- C++ program 2 (tokenized)

        return self.embedding(x) * math.sqrt(self.d_model)
        #
        # STEP 1: self.embedding(x)
        #   For each token ID, look up its row in the embedding table.
        #   Input shape:  (batch_size, seq_len)        = (2, 5)
        #   Output shape: (batch_size, seq_len, d_model) = (2, 5, 512)
        #   Each of those 5 integers became a vector of 512 numbers.
        #
        # STEP 2: * math.sqrt(self.d_model)
        #   Multiply EVERY number by sqrt(512) ≈ 22.6
        #
        #   WHY? This is from the paper, section 3.4.
        #   - Embedding values are initialized with mean≈0, variance≈1
        #     so typical values are small: [-1, 1]
        #   - Positional encoding values (added next) are also in [-1, 1]
        #   - If we don't scale up embeddings, the position signal is just as
        #     strong as the meaning signal — the model can't tell them apart!
        #   - Scaling by √512 ≈ 22.6 makes the word meaning MUCH larger
        #     than the position signal, so position is a gentle hint, not noise.


# =============================================================================
# COMPONENT 2: POSITIONAL ENCODING
# =============================================================================
# PURPOSE: Give the model a sense of token order.
#
# PROBLEM: InputEmbeddings give the same vector to token 34 whether it appears
#          at position 0 or position 99. Transformers have no recurrence or
#          convolution, so without this, the model sees a BAG of tokens with
#          no order — "int x = 5" and "5 = x int" look identical.
#
# SOLUTION: Before feeding embeddings into the transformer, ADD a fixed
#           position-specific vector to each embedding. Same token, different
#           position → different final vector.
#
# THE FORMULA (from paper, section 3.5):
#   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
#
#   pos = position in sequence: 0, 1, 2, ..., seq_len-1
#   i   = dimension pair index: 0, 1, 2, ..., d_model/2 - 1
#         even dimensions (0,2,4,...) use sin
#         odd  dimensions (1,3,5,...) use cos
#
# ANALOGY: Think of a clock with many hands.
#   - The "seconds hand" completes a full rotation every 60 seconds.
#   - The "minutes hand" completes one every 3600 seconds.
#   - The "hours hand" completes one every 43200 seconds.
#   Together they uniquely identify any moment in time.
#   Similarly, the sin/cos waves at different frequencies uniquely
#   identify any position up to very long sequences.
#
# KEY PROPERTY: The encoding is FIXED (not learned). It's computed once
#               and reused. The model just needs to read it.
# =============================================================================


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        # d_model:  same as InputEmbeddings — size of each vector (512)
        # seq_len:  maximum sequence length we'll ever see (e.g. 512 tokens)
        #           We pre-compute PE for ALL positions up to this limit.
        # dropout:  fraction of values randomly zeroed during training (e.g. 0.1)
        #           Adds regularization — prevents over-relying on exact positions.

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # nn.Dropout randomly zeros elements with probability `dropout` during
        # training. During eval (model.eval()), it becomes a no-op.

        # ------------------------------------------------------------------
        # BUILD THE POSITIONAL ENCODING MATRIX
        # Shape: (seq_len, d_model)  e.g. (512, 512)
        # Row i = the encoding vector for position i
        # ------------------------------------------------------------------

        pe = torch.zeros(seq_len, d_model)
        # Allocate an empty (seq_len x d_model) matrix filled with zeros.
        # We'll fill it with sin/cos values below.

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # torch.arange(0, seq_len): creates [0, 1, 2, ..., seq_len-1]  shape: (seq_len,)
        # .unsqueeze(1):            reshapes to [[0],[1],[2],...]        shape: (seq_len, 1)
        # We need it as a column vector so it broadcasts correctly with
        # the row vector of dimension indices below.

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        # This computes 1 / 10000^(2i/d_model) for i = 0, 1, 2, ..., d_model/2-1
        #
        # STEP BY STEP:
        #   torch.arange(0, d_model, 2)  → [0, 2, 4, ..., 510]   shape: (d_model/2,)
        #                                   these are the 2i values
        #
        #   * (-math.log(10000.0) / d_model)
        #                                → [0, -2*log(10000)/512, -4*log(10000)/512, ...]
        #
        #   torch.exp(...)               → [e^0, e^(-2log(10000)/512), ...]
        #                                = [1,   10000^(-2/512), ...]
        #                                = [1/10000^0, 1/10000^(2/512), ...]
        #
        # WHY exp(log(...)) instead of direct division?
        #   Numerical stability. 10000^(510/512) ≈ 9961 which is large.
        #   Computing in log space keeps values well-behaved.

        pe[:, 0::2] = torch.sin(position * div_term)
        # 0::2 = every even column: 0, 2, 4, ..., 510
        # position * div_term: (seq_len,1) * (d_model/2,) → (seq_len, d_model/2)
        #   broadcasting: each position value multiplied by each div_term value
        # sin(...): apply sin element-wise
        # Result fills all even columns of pe.

        pe[:, 1::2] = torch.cos(position * div_term)
        # 1::2 = every odd column: 1, 3, 5, ..., 511
        # Same computation but with cos.
        # Result fills all odd columns of pe.

        pe = pe.unsqueeze(0)
        # Shape was: (seq_len, d_model)
        # Shape now: (1, seq_len, d_model)
        #
        # WHY? Our input embeddings have shape (batch_size, seq_len, d_model).
        # By adding a batch dimension of size 1, PyTorch will automatically
        # BROADCAST pe across all items in the batch — same position encoding
        # applied to every sequence in the batch.

        self.register_buffer('pe', pe)
        # register_buffer tells PyTorch:
        #   "This tensor is part of the module, but it's NOT a trainable parameter."
        #   - It will be saved/loaded with the model (model.state_dict())
        #   - It will move to GPU when you call model.to('cuda')
        #   - But the optimizer will NOT update it during training
        # If we used self.pe = pe instead, it would NOT move to GPU automatically.

    def forward(self, x):
        # x: output from InputEmbeddings
        #    Shape: (batch_size, seq_len, d_model)

        x = x + self.pe[:, :x.shape[1], :]
        # self.pe shape:          (1, seq_len_max, d_model)
        # self.pe[:, :x.shape[1], :] slices to (1, current_seq_len, d_model)
        #   WHY slice? The input sequence might be shorter than seq_len_max.
        #   We only add the positional encodings for the positions that exist.
        #
        # + broadcasts:  (batch_size, seq_len, d_model)
        #              + (1,          seq_len, d_model)
        #              = (batch_size, seq_len, d_model)
        #
        # .requires_grad_(False) is implicitly handled by register_buffer —
        # gradients won't flow into pe during backprop.

        return self.dropout(x)
        # Apply dropout and return.
        # Shape unchanged: (batch_size, seq_len, d_model)


# =============================================================================
# COMPONENT 3: LAYER NORMALIZATION
# =============================================================================
# PURPOSE: Stabilize training by keeping activations at a consistent scale
#          after every sublayer (attention and feed-forward).
#
# THE PROBLEM:
#   Imagine a vector for token "int" after attention: [450, -800, 1200, ...]
#   And another for token "=" after attention:        [0.001, -0.003, 0.002, ...]
#   These wildly different scales make it hard for the next layer to learn —
#   gradients become huge or vanish, and training diverges.
#
# THE SOLUTION — normalize each token's vector independently:
#   1. Compute mean and variance across the 512 dimensions of that one vector.
#   2. Subtract mean, divide by std → values now centered around 0, scale ~1.
#   3. Apply learned scale (gamma) and shift (beta) so the model can
#      adjust the distribution if normalization is too aggressive.
#
# LAYER NORM vs BATCH NORM:
#   Batch Norm: normalizes across the BATCH dimension (across different samples).
#               Problematic for sequences — batch size varies, short sequences
#               have too few samples to estimate good statistics.
#   Layer Norm: normalizes across the FEATURE dimension (across d_model=512).
#               Each token is normalized independently — no dependency on batch.
#               Works perfectly for sequences of any length and batch size.
#
# WHERE IT'S USED: After every attention sublayer and feed-forward sublayer.
#   output = LayerNorm(x + Sublayer(x))   <- the "Add & Norm" block
# =============================================================================


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 1e-6):
        # features: the dimension being normalized — same as d_model (512).
        #           We need this to create gamma and beta of the right size.
        #
        # eps (epsilon): a tiny number added to the denominator to prevent
        #                division by zero when variance is 0.
        #                Default 1e-6 = 0.000001. Never set to 0.

        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(features))
        # alpha = gamma in the paper = the SCALE parameter.
        #
        # nn.Parameter wraps a tensor and tells PyTorch:
        #   "This is a trainable weight — include it in model.parameters()
        #    so the optimizer updates it during training."
        #
        # torch.ones(features): initialized to all 1s → shape: (512,)
        # Why 1? At initialization, we want the output to equal the normalized
        # input (no scaling). The model will learn to deviate from 1 if needed.
        #
        # One alpha value per dimension — the model learns a different scale
        # for each of the 512 dimensions.

        self.bias = nn.Parameter(torch.zeros(features))
        # bias = beta in the paper = the SHIFT parameter.
        #
        # torch.zeros(features): initialized to all 0s → shape: (512,)
        # Why 0? At initialization, no shift. The model learns shifts if needed.
        #
        # One bias value per dimension — the model learns a different shift
        # for each of the 512 dimensions.

    def forward(self, x):
        # x: any tensor with last dimension = features (d_model)
        #    Typical shape: (batch_size, seq_len, d_model) = (2, 5, 512)

        mean = x.mean(dim=-1, keepdim=True)
        # dim=-1: compute mean across the LAST dimension (d_model = 512).
        #         For each token, average its 512 values into one number.
        #
        # keepdim=True: keep the dimension so shapes broadcast correctly.
        #   Without keepdim: shape (2, 5, 512) -> (2, 5)
        #   With    keepdim: shape (2, 5, 512) -> (2, 5, 1)  <- we want this
        #
        # Example for one token: [450, -800, 1200, 0, -50, ...]
        #   mean = (450 + (-800) + 1200 + 0 + (-50) + ...) / 512

        std = x.std(dim=-1, keepdim=True)
        # Compute standard deviation across d_model for each token.
        # Shape: (batch_size, seq_len, 1)
        #
        # std measures how spread out the values are.
        # Large std -> values are far from mean (e.g. -800 to 1200)
        # Small std -> values cluster tightly around mean

        x_norm = (x - mean) / (std + self.eps)
        # Normalize: subtract mean, divide by std.
        #
        # (x - mean): center the values around 0
        #             Shape: (2, 5, 512) - (2, 5, 1) = (2, 5, 512)  [broadcasts]
        #
        # / (std + eps): scale so values have std ≈ 1
        #                + eps prevents division by zero if all values are identical
        #
        # Result: each token's 512 values are now roughly in range [-3, 3]

        return self.alpha * x_norm + self.bias
        # Apply the learned scale (alpha) and shift (bias).
        #
        # self.alpha shape: (512,)   broadcasts to (2, 5, 512)
        # self.bias  shape: (512,)   broadcasts to (2, 5, 512)
        #
        # WHY? Pure normalization might be too restrictive.
        # If the model needs values in a different range for the next layer,
        # alpha and bias let it restore expressiveness.
        # At init: alpha=1, bias=0 -> output = x_norm (pure normalization).
        # After training: alpha and bias are tuned per dimension.


# =============================================================================
# COMPONENT 4: FEED-FORWARD NETWORK (FFN)
# =============================================================================
# PURPOSE: After attention decides WHICH tokens to look at, the FFN decides
#          WHAT to do with that combined information.
#
# ANALOGY:
#   Attention  = a researcher gathering relevant papers from a library
#   FFN        = the researcher sitting down and actually THINKING about
#                what those papers mean for the current question
#
# STRUCTURE (from paper, section 3.3):
#   FFN(x) = max(0, x W1 + b1) W2 + b2
#
#   Linear(d_model -> d_ff)   expand:    512  -> 2048  (4x wider)
#   ReLU                      activate:  adds non-linearity
#   Dropout                   regularize: randomly zero some values
#   Linear(d_ff -> d_model)   contract:  2048 -> 512   (back to original)
#
# WHY EXPAND TO 4x?
#   The wider middle layer gives the model more "working memory" to perform
#   complex transformations. d_ff = 4 * d_model is the paper's choice (2048).
#   Think of it as: the model temporarily uses a bigger scratchpad,
#   does its thinking, then summarizes back down to 512 numbers.
#
# APPLIED PER POSITION:
#   The FFN is applied to each token's vector INDEPENDENTLY and IDENTICALLY.
#   Same weights for every position — but each token gets its own calculation.
#   Shape in:  (batch, seq_len, d_model)
#   Shape out: (batch, seq_len, d_model)   <- unchanged
#
# CONTRAST WITH ATTENTION:
#   Attention:  tokens COMMUNICATE with each other (cross-token operation)
#   FFN:        each token thinks for ITSELF (per-token operation)
# =============================================================================


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        # d_model:  input/output dimension (512)
        # d_ff:     inner (hidden) dimension (2048 = 4 * d_model in the paper)
        # dropout:  fraction of neurons randomly zeroed during training

        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        # First linear layer: expands from d_model to d_ff.
        # nn.Linear(in, out) creates weight matrix W of shape (out, in)
        # and bias vector b of shape (out,).
        # Computation: output = x @ W.T + b
        # Shape: (batch, seq_len, 512) -> (batch, seq_len, 2048)

        self.dropout = nn.Dropout(dropout)
        # Dropout applied after ReLU (between the two linear layers).
        # During training: randomly sets some values to 0.
        # During eval:     does nothing (pass-through).

        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        # Second linear layer: contracts from d_ff back to d_model.
        # Shape: (batch, seq_len, 2048) -> (batch, seq_len, 512)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)

        x = self.linear_1(x)
        # Expand to d_ff.
        # Shape: (batch, seq_len, 512) -> (batch, seq_len, 2048)

        x = torch.relu(x)
        # ReLU: max(0, x) — every negative value becomes 0.
        #
        # WHY ReLU?
        #   Without activation, stacking two linear layers collapses to one:
        #   W2(W1 x) = (W2 W1) x. ReLU breaks this, enabling complex learning.
        #
        # WHY NOT sigmoid/tanh?
        #   ReLU doesn't saturate for positive values — gradients flow cleanly.
        #   Sigmoid/tanh squash large values, killing gradients (vanishing gradient).

        x = self.dropout(x)
        # Dropout after activation, before next linear layer.

        x = self.linear_2(x)
        # Contract back to d_model.
        # Shape: (batch, seq_len, 2048) -> (batch, seq_len, 512)

        return x
        # Output shape: (batch_size, seq_len, d_model) — same as input.


# =============================================================================
# COMPONENT 5: MULTI-HEAD ATTENTION
# =============================================================================
# PURPOSE: Let every token look at every other token and decide how much
#          attention to pay to each one. This is HOW tokens communicate.
#
# STRUCTURE:
#   1. Project Q, K, V with learned weight matrices
#   2. Split into h=8 heads (each head works in 64-dim subspace)
#   3. Run scaled dot-product attention independently per head
#   4. Concatenate all head outputs
#   5. Project through W_O to mix information across heads
#
# THREE USES (same module, different inputs):
#   Encoder self-attention:  q=k=v=encoder output  (tokens look at each other)
#   Decoder self-attention:  q=k=v=decoder output  (masked: can't see future)
#   Cross-attention:         q=decoder, k=v=encoder (decoder looks at encoder)
# =============================================================================


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        # d_model: embedding dimension (512)
        # h:       number of attention heads (8 in the paper)
        # dropout: applied to attention weights

        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        # d_k = 64: dimension per head (512 / 8 = 64)
        # Each head works in a 64-dim subspace of the full 512-dim space.

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Projects input into Query space.
        # Internally holds ALL 8 heads' W_Qi matrices packed into one (512x512).
        # bias=False: standard in attention projections.

        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Projects input into Key space.

        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Projects input into Value space.

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        # Output projection W_O: applied after concatenating all heads.
        # This is where heads communicate — W_O mixes all head outputs.

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        # Scaled dot-product attention.
        # query, key, value shape: (batch, h, seq_len, d_k)
        # mask: optional, blocks certain positions from being attended to

        d_k = query.shape[-1]
        # d_k = 64. Needed for the sqrt scaling.

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # STEP 1: Raw attention scores.
        #
        # key.transpose(-2, -1):
        #   (batch, h, seq_len, d_k) -> (batch, h, d_k, seq_len)
        #
        # query @ key.transpose(-2, -1):
        #   (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len)
        #   = (batch, h, seq_len, seq_len)
        #
        #   score[b, head, i, j] = how much token i should attend to token j
        #
        # / sqrt(d_k): prevents dot products from getting too large,
        #   which would push softmax into saturation and kill gradients.

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            # Set forbidden positions to -infinity before softmax.
            # After softmax, e^(-1e9) ~= 0 -> those positions get ~0 weight.
            #
            # Padding mask:  ignores [PAD] tokens added to equalise batch lengths.
            # Causal mask:   token i cannot see tokens at positions j > i (future).
            #                Used in decoder so it can't cheat during training.

        attention_scores = attention_scores.softmax(dim=-1)
        # STEP 2: Convert scores to probabilities.
        # Each row sums to 1. High score -> token gets more attention.
        # Shape unchanged: (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Randomly zero some attention weights during training.
        # Prevents over-relying on specific attention patterns.

        output = attention_scores @ value
        # STEP 3: Weighted sum of values.
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k)
        # = (batch, h, seq_len, d_k)
        #
        # output[i] = sum over j of (attention_weight[i,j] * value[j])
        # Each token's output is a blend of all tokens' values,
        # weighted by how much attention it paid to each.

        return output, attention_scores
        # Return attention_scores too — useful for visualisation/debugging.

    def forward(self, q, k, v, mask):
        # q, k, v shape: (batch_size, seq_len, d_model)
        # mask: (batch_size, 1, seq_len, seq_len) or None

        query = self.w_q(q)
        # Project to query space: (batch, seq_len, 512)

        key = self.w_k(k)
        # Project to key space: (batch, seq_len, 512)

        value = self.w_v(v)
        # Project to value space: (batch, seq_len, 512)

        # --- Split into h heads ---
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        # .view: reshape last dim 512 -> (8, 64)
        #   (batch, seq_len, 512) -> (batch, seq_len, 8, 64)
        # .transpose(1, 2): move head dim forward
        #   (batch, seq_len, 8, 64) -> (batch, 8, seq_len, 64)
        # Now each head's slice is a (seq_len, 64) matrix, independent of others.

        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # Same split for key and value.

        # --- Run attention on all 8 heads in parallel ---
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # x shape: (batch, 8, seq_len, 64)
        # All 8 heads computed simultaneously — no loop needed.

        # --- Concatenate heads ---
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # .transpose(1, 2): move seq_len back before head dim
        #   (batch, 8, seq_len, 64) -> (batch, seq_len, 8, 64)
        # .contiguous(): required before .view() after a transpose
        # .view(batch, -1, 512): merge head and d_k dims back together
        #   (batch, seq_len, 8, 64) -> (batch, seq_len, 512)

        return self.w_o(x)
        # Output projection: mix information across all heads.
        # (batch, seq_len, 512) -> (batch, seq_len, 512)
        # This is the only place where the 8 heads communicate with each other.


# =============================================================================
# COMPONENT 6: RESIDUAL CONNECTION
# =============================================================================
# PURPOSE: Allow gradients to flow through deep networks without vanishing,
#          and let the model learn corrections rather than full transformations.
#
# THE PROBLEM IT SOLVES:
#   A transformer encoder has 6 stacked layers. Without skip connections,
#   gradients must pass through all 6 layers during backprop, multiplying
#   by small numbers each time. By layer 1, the gradient is near zero —
#   early layers stop learning entirely (vanishing gradient problem).
#
# THE SOLUTION:
#   output = x + sublayer(LayerNorm(x))
#
#   Gradients now have two paths back:
#     Path 1: through the sublayer (can be small)
#     Path 2: directly through +x (gradient = 1, always clean)
#
# PRE-LN vs POST-LN:
#   Original paper (Post-LN): output = LayerNorm(x + sublayer(x))
#   Modern practice (Pre-LN): output = x + sublayer(LayerNorm(x))
#   We use Pre-LN — more stable, doesn't need learning rate warmup tricks.
#
# USED TWICE PER ENCODER BLOCK:
#   x = x + Attention(LayerNorm(x))   <- after self-attention
#   x = x + FFN(LayerNorm(x))         <- after feed-forward
# =============================================================================


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float):
        # features: d_model (512) — passed to LayerNormalization
        # dropout:  applied to sublayer output before adding residual

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        # Each ResidualConnection owns its own LayerNorm instance.
        # An EncoderBlock creates two ResidualConnections (one for attention,
        # one for FFN), each with independently learned norm parameters.

    def forward(self, x, sublayer):
        # x:        input tensor, shape (batch, seq_len, d_model)
        # sublayer: any callable — attention or FFN — passed in at call time.
        #           This makes ResidualConnection reusable for both without
        #           needing separate subclasses.

        return x + self.dropout(sublayer(self.norm(x)))
        # 1. self.norm(x)         — LayerNorm the input (Pre-LN)
        # 2. sublayer(...)        — pass through attention or FFN
        # 3. self.dropout(...)    — regularisation on sublayer output
        # 4. x + ...              — skip connection: add original input back
        # Shape unchanged throughout: (batch, seq_len, d_model)


# =============================================================================
# COMPONENT 7: ENCODER BLOCK
# =============================================================================
# PURPOSE: One complete encoder layer. The Encoder stacks N=6 of these.
#
# STRUCTURE:
#
#   Input x (batch, seq_len, 512)
#       |
#       v
#   [ResidualConnection 1]
#       x = x + MultiHeadAttention(LayerNorm(x))
#       "Tokens look at each other and gather context"
#       |
#       v
#   [ResidualConnection 2]
#       x = x + FFN(LayerNorm(x))
#       "Each token thinks about what it learned from attention"
#       |
#       v
#   Output x (batch, seq_len, 512)   <- same shape, richer representation
#
# WHAT CHANGES WITH EACH BLOCK?
#   The vectors for each token get progressively richer.
#   Block 1 might learn: basic syntax (this token is a type keyword)
#   Block 2 might learn: local relationships (this variable was declared here)
#   Block 3 might learn: scope (this variable is used inside this function)
#   ...and so on, building up increasingly abstract understanding.
#
# NOTE — ENCODER ONLY HAS SELF-ATTENTION:
#   q = k = v = x (the same sequence attends to itself).
#   The decoder will have TWO attention sublayers (self + cross-attention),
#   which is why we build EncoderBlock and DecoderBlock separately.
# =============================================================================


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        # features:             d_model (512)
        # self_attention_block: a pre-built MultiHeadAttentionBlock instance
        # feed_forward_block:   a pre-built FeedForwardBlock instance
        # dropout:              passed to both ResidualConnections
        #
        # WHY pass pre-built blocks instead of building them here?
        #   Flexibility. The factory function (build_transformer) will construct
        #   all blocks with the right hyperparameters and inject them here.
        #   This keeps EncoderBlock clean and focused on assembly only.

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block   = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )
        # nn.ModuleList: a list of nn.Module instances that PyTorch tracks.
        # We need TWO residual connections per encoder block:
        #   [0] wraps self-attention
        #   [1] wraps FFN
        #
        # WHY nn.ModuleList and not a plain Python list?
        #   A plain list [] is invisible to PyTorch — its parameters won't
        #   be included in model.parameters() and won't be trained.
        #   nn.ModuleList properly registers all contained modules.

    def forward(self, x, src_mask):
        # x:        (batch, seq_len, d_model)
        # src_mask: padding mask for the source sequence.
        #           Prevents attention to [PAD] tokens added to equalise
        #           sequence lengths within a batch.
        #           Shape: (batch, 1, 1, seq_len) — broadcasts over heads and queries.

        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        # SUBLAYER 1: Self-attention wrapped in residual connection.
        #
        # lambda x: self.self_attention_block(x, x, x, src_mask)
        #   Creates a function that takes x and runs self-attention on it.
        #   q = k = v = x because this is SELF-attention (tokens attend to themselves).
        #   The lambda lets ResidualConnection call sublayer(norm(x)) internally,
        #   where norm(x) gets passed as the q, k, and v simultaneously.
        #
        # After this line:
        #   x = x + Attention(LayerNorm(x))

        x = self.residual_connections[1](x, self.feed_forward_block)
        # SUBLAYER 2: FFN wrapped in residual connection.
        #
        # feed_forward_block is passed directly (no lambda needed) because
        # FFN takes exactly one argument x, matching sublayer(norm(x)) perfectly.
        #
        # After this line:
        #   x = x + FFN(LayerNorm(x))

        return x
        # Output shape: (batch, seq_len, d_model) — unchanged.
        # Content: each token now contains context gathered from all other tokens
        # via attention, processed through the FFN.


# =============================================================================
# COMPONENT 8: ENCODER
# =============================================================================
# PURPOSE: Stack N EncoderBlocks to build progressively richer representations
#          of the C++ source sequence.
#
# STRUCTURE:
#   Input x
#     -> EncoderBlock 1  (basic syntax understanding)
#     -> EncoderBlock 2  (local relationships)
#     -> EncoderBlock 3  (scope and dependencies)
#     -> EncoderBlock 4  (semantics)
#     -> EncoderBlock 5  (patterns and idioms)
#     -> EncoderBlock 6  (high-level intent)
#     -> LayerNorm       (final normalisation)
#     -> output sent to decoder via cross-attention
#
# WHY A FINAL LAYERNORM?
#   After 6 blocks of residual additions, the final output can drift in scale.
#   One last LayerNorm stabilises the encoder output before the decoder reads it.
#   This is standard in Pre-LN transformer implementations.
#
# FOR OUR PROJECT:
#   The encoder reads "int x = 5 ;" (C++ tokens) and produces a rich
#   context vector for each token. The decoder then uses these vectors
#   via cross-attention to generate the Python equivalent "x = 5".
# =============================================================================


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        # features: d_model (512)
        # layers:   nn.ModuleList of N EncoderBlock instances (typically N=6)

        super().__init__()
        self.layers = layers
        # Store the list of EncoderBlocks.
        # nn.ModuleList ensures all blocks and their parameters are
        # properly registered and moved to GPU with .to(device).

        self.norm = LayerNormalization(features)
        # Final LayerNorm applied after all blocks.
        # Stabilises the encoder output before it goes to the decoder.

    def forward(self, x, mask):
        # x:    (batch, seq_len, d_model) — embedded + positionally encoded tokens
        # mask: padding mask for the C++ source sequence

        for layer in self.layers:
            x = layer(x, mask)
        # Pass x through each EncoderBlock one by one.
        # Each block refines the representation:
        #   x after block 1: tokens aware of immediate neighbours
        #   x after block 2: tokens aware of broader context
        #   ...
        #   x after block 6: tokens carry full sequence-level understanding
        # Shape stays (batch, seq_len, d_model) throughout — only content changes.

        return self.norm(x)
        # Apply final LayerNorm and return.
        # Output shape: (batch, seq_len, d_model)
        # This output will be used as k and v in the decoder's cross-attention.


# =============================================================================
# COMPONENT 9: DECODER BLOCK
# =============================================================================
# PURPOSE: One complete decoder layer. Generates Python tokens one at a time
#          by attending to (1) previously generated Python tokens and
#          (2) the full C++ encoder output.
#
# STRUCTURE (3 sublayers vs encoder's 2):
#
#   Input x  (Python tokens generated so far)
#       |
#       v
#   [ResidualConnection 1] — Masked Self-Attention
#       x = x + MaskedSelfAttention(LayerNorm(x), mask=tgt_mask)
#       "Each Python token looks at all PREVIOUS Python tokens only."
#       tgt_mask is causal: token i cannot see tokens i+1, i+2, ...
#       Without this mask, the decoder would see the answer during training.
#       |
#       v
#   [ResidualConnection 2] — Cross-Attention
#       x = x + CrossAttention(LayerNorm(x), encoder_output, encoder_output, src_mask)
#       q = decoder state    ("what Python token am I generating right now?")
#       k = encoder output   ("what C++ tokens are available to look at?")
#       v = encoder output   ("what information do those C++ tokens carry?")
#       "Each Python token asks: which parts of the C++ input matter to me?"
#       |
#       v
#   [ResidualConnection 3] — FFN
#       x = x + FFN(LayerNorm(x))
#       "Process and transform what was gathered from both attention layers."
#       |
#       v
#   Output (batch, tgt_seq_len, d_model)
#
# TWO MASKS:
#   tgt_mask: causal mask — Python token i cannot see Python tokens j > i
#   src_mask: padding mask — prevents attending to [PAD] tokens in C++ input
# =============================================================================


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        # features:              d_model (512)
        # self_attention_block:  MultiHeadAttentionBlock for masked self-attention
        # cross_attention_block: MultiHeadAttentionBlock for cross-attention
        #                        Separate instance from self_attention_block —
        #                        each has its own independently learned weights.
        # feed_forward_block:    FeedForwardBlock instance
        # dropout:               passed to all three ResidualConnections

        super().__init__()
        self.self_attention_block  = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block    = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )
        # THREE residual connections — one per sublayer:
        # [0] -> masked self-attention
        # [1] -> cross-attention
        # [2] -> FFN

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x:              (batch, tgt_seq_len, d_model) — Python tokens so far
        # encoder_output: (batch, src_seq_len, d_model) — C++ context from encoder
        # src_mask:       padding mask for C++ source (blocks [PAD] tokens)
        # tgt_mask:       causal mask for Python target (blocks future tokens)

        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # SUBLAYER 1: Masked self-attention.
        # q = k = v = x  (Python tokens attend to each other).
        # tgt_mask ensures token i only sees tokens 0..i.
        # After this: each Python token carries context from previous Python tokens.

        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        # SUBLAYER 2: Cross-attention.
        # q = x (decoder)              — what am I generating?
        # k = v = encoder_output (C++) — what C++ context is available?
        # src_mask blocks [PAD] in C++ input.
        # After this: each Python token knows which C++ tokens to focus on.

        x = self.residual_connections[2](x, self.feed_forward_block)
        # SUBLAYER 3: FFN.
        # Same as encoder — processes what both attention layers gathered.

        return x
        # Output shape: (batch, tgt_seq_len, d_model)


# =============================================================================
# COMPONENT 10: DECODER
# =============================================================================
# PURPOSE: Stack N DecoderBlocks to progressively build the Python output,
#          guided at every layer by the C++ encoder output via cross-attention.
#
# STRUCTURE:
#   Python tokens so far
#     -> DecoderBlock 1  (basic token relationships + first look at C++)
#     -> DecoderBlock 2  (local Python syntax + refined C++ attention)
#     -> DecoderBlock 3  (scope + variable mapping from C++ to Python)
#     -> DecoderBlock 4  (semantics of what's being translated)
#     -> DecoderBlock 5  (idiomatic Python patterns)
#     -> DecoderBlock 6  (high-level intent of the translation)
#     -> LayerNorm       (final stabilisation)
#     -> output fed into ProjectionLayer to predict next Python token
#
# KEY DIFFERENCE FROM ENCODER:
#   forward() takes FOUR arguments instead of two:
#     x, encoder_output, src_mask, tgt_mask
#   encoder_output and src_mask are passed through to every DecoderBlock
#   so every layer can attend to the full C++ context via cross-attention.
# =============================================================================


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        # features: d_model (512)
        # layers:   nn.ModuleList of N DecoderBlock instances (typically N=6)

        super().__init__()
        self.layers = layers
        self.norm   = LayerNormalization(features)
        # Final LayerNorm — same reason as in the Encoder:
        # stabilises the output after 6 rounds of residual additions.

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x:              (batch, tgt_seq_len, d_model) — Python tokens so far
        # encoder_output: (batch, src_seq_len, d_model) — frozen C++ context
        # src_mask:       padding mask for C++ source
        # tgt_mask:       causal mask for Python target

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Pass through each DecoderBlock one by one.
        # encoder_output is the SAME at every layer — it's the fixed C++ context
        # computed once by the encoder. Each decoder layer attends to it freshly
        # via its own cross-attention weights, learning different aspects of
        # the C++ input at different levels of abstraction.

        return self.norm(x)
        # Apply final LayerNorm and return.
        # Output shape: (batch, tgt_seq_len, d_model)
        # This feeds into the ProjectionLayer to produce vocabulary probabilities.


# =============================================================================
# COMPONENT 11: PROJECTION LAYER
# =============================================================================
# PURPOSE: Convert the decoder's 512-dim output vectors into a probability
#          distribution over the Python vocabulary — so we can pick the
#          next Python token to generate.
#
# THE PROBLEM:
#   The decoder outputs (batch, tgt_seq_len, 512) — 512 numbers per position.
#   These numbers are meaningful to the model internally, but we need to
#   answer: "what is the next Python token?" from a vocabulary of ~8000 tokens.
#
# THE SOLUTION:
#   A linear layer maps 512 -> vocab_size, giving a raw score (logit) for
#   every token in the vocabulary. Then log_softmax converts to log-probabilities.
#
#   Example for one position:
#     512 numbers -> Linear -> [score_"x", score_"=", score_"5", score_"for", ...]
#                                vocab_size scores (e.g. 8000)
#                 -> log_softmax -> [log_prob_"x", log_prob_"=", ...]
#                                   all values <= 0, sum of exp = 1.0
#
# WHY LOG_SOFTMAX INSTEAD OF SOFTMAX?
#   During training we use NLLLoss (negative log-likelihood):
#     loss = -log_prob[correct_token]
#   log_softmax + NLLLoss is mathematically equivalent to CrossEntropyLoss
#   but more numerically stable than computing softmax then log separately.
#
# AT INFERENCE:
#   Take argmax over vocab_size at the last position to get predicted token:
#     next_token = argmax(projection_output[:, -1, :])
# =============================================================================


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        # d_model:    input dimension (512)
        # vocab_size: number of tokens in the Python vocabulary

        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        # Weight shape: (vocab_size, d_model) e.g. (8000, 512)
        # Maps each 512-dim decoder output to a score for every Python token.

    def forward(self, x):
        # x shape: (batch, tgt_seq_len, d_model)

        return torch.log_softmax(self.proj(x), dim=-1)
        # STEP 1: self.proj(x)
        #   Linear projection: (batch, tgt_seq_len, 512) -> (batch, tgt_seq_len, vocab_size)
        #   Each position now has one raw score per vocabulary token.
        #
        # STEP 2: torch.log_softmax(..., dim=-1)
        #   Apply log_softmax across the vocabulary dimension (last dim).
        #   Converts raw scores to log-probabilities.
        #   All values <= 0. The highest value = most likely next token.
        #
        # Output shape: (batch, tgt_seq_len, vocab_size)


# =============================================================================
# COMPONENT 12A: TRANSFORMER
# =============================================================================
# PURPOSE: The top-level container that holds all components and exposes
#          three clean methods: encode(), decode(), project().
#
# WHY THREE SEPARATE METHODS INSTEAD OF ONE forward()?
#   During inference, we generate one token at a time:
#     1. encode() is called ONCE on the full C++ input
#     2. decode() is called N times (once per generated Python token)
#     3. project() is called N times to get vocabulary probabilities
#
#   If we had one forward(), we'd have to re-run the encoder every step,
#   which is wasteful. Separating them lets us cache encoder_output.
#
# FULL PIPELINE:
#   C++ tokens  -> src_embed -> src_pos -> encoder -> encoder_output
#   Python tokens -> tgt_embed -> tgt_pos -> decoder(encoder_output) -> projection -> log_probs
# =============================================================================


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):

        super().__init__()
        self.encoder          = encoder
        self.decoder          = decoder
        self.src_embed        = src_embed    # embedding for C++ tokens
        self.tgt_embed        = tgt_embed    # embedding for Python tokens
        self.src_pos          = src_pos      # positional encoding for C++
        self.tgt_pos          = tgt_pos      # positional encoding for Python
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # src:      (batch, src_seq_len)  — C++ token IDs
        # src_mask: padding mask for C++ input
        src = self.src_embed(src)    # token IDs -> 512-dim vectors
        src = self.src_pos(src)      # add positional encoding
        return self.encoder(src, src_mask)
        # Output: (batch, src_seq_len, d_model) — rich C++ context vectors

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # encoder_output: (batch, src_seq_len, d_model) — from encode()
        # src_mask:       padding mask for C++ source
        # tgt:            (batch, tgt_seq_len) — Python token IDs generated so far
        # tgt_mask:       causal mask for Python target
        tgt = self.tgt_embed(tgt)    # token IDs -> 512-dim vectors
        tgt = self.tgt_pos(tgt)      # add positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        # Output: (batch, tgt_seq_len, d_model)

    def project(self, x):
        # x: (batch, tgt_seq_len, d_model) — from decode()
        return self.projection_layer(x)
        # Output: (batch, tgt_seq_len, vocab_size) — log-probabilities


# =============================================================================
# COMPONENT 12B: BUILD TRANSFORMER
# =============================================================================
# PURPOSE: Factory function that constructs a complete Transformer with all
#          hyperparameters, initialises weights, and returns it ready to train.
#
# WEIGHT INITIALISATION — Xavier Uniform:
#   By default PyTorch initialises weights randomly from a small uniform dist.
#   Xavier (Glorot) uniform is better for deep networks:
#     W ~ Uniform(-sqrt(6 / (fan_in + fan_out)), +sqrt(6 / (fan_in + fan_out)))
#   It keeps the variance of activations consistent across layers,
#   preventing vanishing/exploding gradients at the start of training.
#   Only applied to matrices (dim > 1) — not to bias vectors or scalars.
# =============================================================================


def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # src_vocab_size: number of unique C++ tokens
    # tgt_vocab_size: number of unique Python tokens
    # src_seq_len:    max C++ sequence length
    # tgt_seq_len:    max Python sequence length
    # d_model:        embedding dimension (default 512, as in paper)
    # N:              number of encoder and decoder blocks (default 6)
    # h:              number of attention heads (default 8)
    # dropout:        dropout rate (default 0.1, as in paper)
    # d_ff:           feed-forward inner dimension (default 2048)

    # --- Embedding layers ---
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # Separate embedding tables for C++ and Python vocabularies.
    # They have different vocab sizes and learn different token representations.

    # --- Positional encoding layers ---
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # Separate instances so each can handle different max sequence lengths.
    # The sin/cos values are fixed (not learned), but dropout differs per call.

    # --- Encoder blocks ---
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward           = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blocks.append(
            EncoderBlock(d_model, encoder_self_attention, feed_forward, dropout)
        )
    # Each block gets its OWN attention and FFN instances with independent weights.
    # They start with the same initialisation but diverge during training.

    # --- Decoder blocks ---
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention  = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward            = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blocks.append(
            DecoderBlock(d_model, decoder_self_attention,
                         decoder_cross_attention, feed_forward, dropout)
        )

    # --- Encoder and Decoder ---
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # --- Projection layer ---
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # --- Assemble Transformer ---
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,
                               src_pos, tgt_pos, projection_layer)

    # --- Initialise weights with Xavier uniform ---
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # p.dim() > 1: skip 1D tensors (biases, LayerNorm alpha/beta).
    # Xavier only makes sense for weight matrices, not vectors.

    return transformer
