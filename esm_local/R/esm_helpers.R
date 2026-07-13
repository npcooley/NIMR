###### -- esm handling functions ----------------------------------------------

###### -- notes ---------------------------------------------------------------
# rely on functions within `torch`, `safetensors`,

###### -- param assignments ---------------------------------------------------
# every single tensor pulled from the checkpoint goes through this function
# rather than direct list indexing -- a typo'd or wrong key name fails loudly,
# with candidate matches, instead of silently producing a NULL/garbage tensor
# that only shows up as a confusing error (or worse, no error at all) later.

# check a parameter name against the names present in the weights
get_param <- function(name,
                      weights) {
  if (!name %in% names(weights)) {
    candidates <- names(weights)[
      agrepl(name, names(weights),
             max.distance = 0.3,
             ignore.case = TRUE)
    ]
    stop("expected weight key not found: '",
         name,
         "'\n",
         "closest matches actually present in the checkpoint:\n",
         paste("  -",
               head(candidates,
                    10),
               collapse = "\n"),
         "\n",
         "inspect `names(weights)` directly and update get_param() calls to match.")
  }
  weights[[name]]
}

###### -- copy function -------------------------------------------------------

copy_into <- function(param,
                      tensor_name,
                      weights) {
  source_tensor <- get_param(tensor_name,
                             weights = weights)
  if (!all(dim(param) == dim(source_tensor))) {
    stop(
      "shape mismatch loading '", tensor_name, "': ",
      "module parameter is ", paste(dim(param), collapse = "x"),
      ", checkpoint tensor is ", paste(dim(source_tensor), collapse = "x")
    )
  }
  with_no_grad({
    param$copy_(source_tensor)
  })
  invisible(NULL)
}

###### -- tokenize character strings ------------------------------------------

# converts one amino acid string into a vector of integer token ids,
# including the leading <cls> and trailing <eos>
tokenize_sequence <- function(aa_string,
                              token_to_id,
                              unk_id) {
  residues <- strsplit(aa_string, "")[[1]]
  ids <- vapply(X = residues,
                FUN = function(residue) {
                  if (residue %in% names(token_to_id)) {
                    token_to_id[[residue]]
                  } else {
                    unk_id
                  }
                },
                FUN.VALUE = vector(mode = "integer",
                                   length = 1))
  res <- c(cls_id,
           ids,
           eos_id)
  return(res)
}

# tokenizes a batch of sequences (character vector), right-pads to the
# longest sequence in the batch, and returns both the integer id tensor
# and an attention mask tensor (1 = real token, 0 = padding)
tokenize_batch <- function(aa_strings,
                           token_to_id,
                           unk_id) {
  token_lists <- lapply(X = aa_strings,
                        FUN = function(x) {
                          tokenize_sequence(x,
                                            token_to_id,
                                            unk_id)
                        })
  max_len <- max(vapply(X = token_lists,
                        FUN = function(x) {
                          length(x)
                        },
                        FUN.VALUE = vector(mode = "integer",
                                           length = 1)))
  
  padded_ids <- matrix(pad_id,
                       nrow = length(token_lists),
                       ncol = max_len)
  attn_mask <- matrix(0L,
                      nrow = length(token_lists),
                      ncol = max_len)
  
  for (i in seq_along(token_lists)) {
    len_i <- length(token_lists[[i]])
    padded_ids[i, 1:len_i] <- token_lists[[i]]
    attn_mask[i, 1:len_i] <- 1L
  }
  
  res <- list("input_ids" = torch_tensor(padded_ids,
                                         dtype = torch_long()),
              "attention_mask" = torch_tensor(attn_mask,
                                              dtype = torch_float()))
  return(res)
}

###### -- rotary position embeddings ------------------------------------------

# standard RoPE, matching the "rotate half" formulation used by HF's
# EsmRotaryEmbedding / apply_rotary_pos_emb -- computed fresh per forward call
# rather than cached, since sequence length varies protein to protein
compute_rotary_cos_sin <- function(seq_len,
                                   dim,
                                   device) {
  inv_freq <- 1 / (10000 ^ ((torch_arange(start = 0,
                                          end = dim - 1,
                                          step = 2,
                                          dtype = torch_float(),
                                          device = device)) / dim))
  positions <- torch_arange(start = 0,
                            end = seq_len - 1,
                            dtype = torch_float(),
                            device = device)
  # (seq_len, dim/2)
  freqs <- torch_outer(positions,
                       inv_freq)
  # (seq_len, dim)
  emb <- torch_cat(list(freqs,
                        freqs),
                   dim = -1)
  # each (seq_len, dim)
  list(cos = emb$cos(),
       sin = emb$sin())
}

rotate_half <- function(x) {
  d <- x$size(-1)
  halves <- torch_split(x,
                        split_size = d / 2,
                        dim = -1)
  res <- torch_cat(list(-halves[[2]],
                        halves[[1]]),
                   dim = -1)
  return(res)
}

apply_rotary <- function(q,
                         k,
                         cos,
                         sin) {
  # q, k: (batch, n_heads, seq_len, head_dim)
  # cos, sin: (seq_len, head_dim) -- reshape for broadcasting
  cos_b <- cos$unsqueeze(1)$unsqueeze(1)   # (1, 1, seq_len, head_dim)
  sin_b <- sin$unsqueeze(1)$unsqueeze(1)
  q_rot <- (q * cos_b) + (rotate_half(q) * sin_b)
  k_rot <- (k * cos_b) + (rotate_half(k) * sin_b)
  res <- list(q = q_rot,
              k = k_rot)
  return(res)
}

###### -- self-attention module -----------------------------------------------

# nn_module is a constructor from `torch`
# function names within it have to match to methods that nn_modules can invoke?
# this is OOP stuff...
# initialize, split_heads, forward ... other methods?
# nn_module requires x = funcion(...) {...}
# as opposed to x <- function(...) {...}
# on its interior

esm_self_attention <- nn_module(
  initialize = function(hidden_size,
                        n_heads) {
    self$hidden_size <- hidden_size
    self$n_heads <- n_heads
    self$head_dim <- hidden_size / n_heads
    
    # nn_linear is more OOP, and produces an R6 obj?
    # so x <- y <- nn_linear(z, w)
    # is not equivalent to:
    # x <- nn_linear(z, w)
    # y <- nn_linear(z, w)
    self$query <- nn_linear(hidden_size,
                            hidden_size)
    self$key <- nn_linear(hidden_size,
                          hidden_size)
    self$value <- nn_linear(hidden_size,
                            hidden_size)
    self$output <- nn_linear(hidden_size,
                             hidden_size)
  },
  split_heads = function(x,
                         batch_size,
                         seq_len) {
    x$view(c(batch_size,
             seq_len,
             self$n_heads,
             self$head_dim))$permute(c(1, 3, 2, 4))
  },
  forward = function(x,
                     additive_mask) {
    batch_size <- x$size(1)
    seq_len <- x$size(2)
    
    q <- self$split_heads(self$query(x),
                          batch_size,
                          seq_len)
    k <- self$split_heads(self$key(x),
                          batch_size,
                          seq_len)
    v <- self$split_heads(self$value(x),
                          batch_size,
                          seq_len)
    
    rotary <- compute_rotary_cos_sin(seq_len,
                                     self$head_dim,
                                     x$device)
    rotated <- apply_rotary(q,
                            k,
                            rotary$cos,
                            rotary$sin)
    q <- rotated$q
    k <- rotated$k
    
    scores <- torch_matmul(q,
                           k$transpose(-1, -2)) / sqrt(self$head_dim)
    # additive_mask broadcasts (batch,1,1,seq_len)
    scores <- scores + additive_mask
    
    probs <- nnf_softmax(scores, dim = -1)
    # (batch, heads, seq_len, head_dim)
    context <- torch_matmul(probs, v)
    context <- context$permute(c(1, 3, 2, 4))$contiguous()$view(c(batch_size,
                                                                  seq_len,
                                                                  self$hidden_size))
    
    self$output(context)
  }
)

###### -- one transformer layer (pre-norm, matching ESM2's architecture) ------

esm_layer <- nn_module(
  initialize = function(hidden_size,
                        n_heads,
                        intermediate_size,
                        eps) {
    self$attention <- esm_self_attention(hidden_size,
                                         n_heads)
    self$pre_attention_ln <- nn_layer_norm(hidden_size,
                                           eps = eps)
    self$intermediate <- nn_linear(hidden_size,
                                   intermediate_size)
    self$ffn_output <- nn_linear(intermediate_size,
                                 hidden_size)
    self$pre_ffn_ln <- nn_layer_norm(hidden_size,
                                     eps = eps)
  },
  forward = function(x,
                     additive_mask) {
    attn_out <- self$attention(self$pre_attention_ln(x),
                               additive_mask)
    x <- x + attn_out
    
    ffn_in  <- self$pre_ffn_ln(x)
    ffn_out <- self$ffn_output(nnf_gelu(self$intermediate(ffn_in)))
    x + ffn_out
  }
)

###### -- encoder things ------------------------------------------------------

esm_encoder <- nn_module(
  initialize = function(vocab_size,
                        hidden_size,
                        n_layers,
                        n_heads,
                        intermediate_size, eps) {
    self$embed_tokens <- nn_embedding(vocab_size,
                                      hidden_size)
    self$layers <- nn_module_list(
      lapply(X = seq_len(n_layers),
             FUN = function(i) {
               esm_layer(hidden_size,
                         n_heads,
                         intermediate_size,
                         eps)
             })
    )
    self$final_ln <- nn_layer_norm(hidden_size,
                                   eps = eps)
  },
  forward = function(input_ids,
                     attention_mask) {
    # additive mask: 0 where real token, large negative where padding,
    # shaped (batch, 1, 1, seq_len) to broadcast against attention scores
    additive_mask <- (1 - attention_mask)$unsqueeze(2)$unsqueeze(3) * -1e9
    
    x <- self$embed_tokens(input_ids + 1L)  # +1: R torch embedding indices are 1-based
    
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x,
                            additive_mask)
    }
    
    self$final_ln(x)
  }
)

###### -- lm head -------------------------------------------------------------

esm_lm_head <- nn_module(
  initialize = function(hidden_size, vocab_size, eps) {
    self$dense <- nn_linear(hidden_size, hidden_size)
    self$layer_norm <- nn_layer_norm(hidden_size, eps = eps)
    self$decoder <- nn_linear(hidden_size, vocab_size, bias = FALSE)
    self$bias <- nn_parameter(torch_zeros(vocab_size))
  },
  forward = function(x) {
    x <- nnf_gelu(self$dense(x))
    x <- self$layer_norm(x)
    self$decoder(x) + self$bias
  }
)

###### -- pseudo-likelihood scores --------------------------------------------

pseudo_likelihood <- function(aa_string,
                              token_to_id,
                              unk_id) {
  base_ids <- tokenize_sequence(aa_string = aa_string,
                                token_to_id = token_to_id,
                                unk_id = unk_id)
  real_len <- nchar(aa_string)
  
  # one row per residue position; row i has position (i+1) replaced with <mask>
  # (+1 in the column index skips the leading <cls>)
  batch_matrix <- matrix(rep(base_ids,
                             each = real_len),
                         nrow = real_len,
                         byrow = TRUE)
  for (i in seq_len(real_len)) {
    batch_matrix[i, i + 1] <- mask_id
  }
  
  input_ids <- torch_tensor(batch_matrix, dtype = torch_long())$to(device = device)
  attn_mask <- torch_ones_like(input_ids, dtype = torch_float())$to(device = device)
  
  with_no_grad({
    hidden  <- model(input_ids, attn_mask)
    logits  <- lm_head(hidden)
  })
  
  log_probs <- nnf_log_softmax(logits, dim = -1)
  true_ids  <- base_ids[2:(real_len + 1)] + 1L   # 1-indexed for logits lookup
  
  position_log_probs <- sapply(seq_len(real_len), function(i) {
    as.numeric(log_probs[i, i + 1, true_ids[i]]$to(device = "cpu"))
  })
  
  return(position_log_probs)
}



