###### -- tempate script ------------------------------------------------------

###### -- libraries -----------------------------------------------------------

library(torch)
library(hfhub)
library(safetensors)
library(Biostrings)
library(jsonlite)

# helper functions
source("esm_helpers.R")

###### -- inputs --------------------------------------------------------------

# hf model
repo <- "facebook/esm2_t12_35M_UR50D"
# data to build an example from
# a genbank submission that was annotated by the submitter:
# https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_900660285.1/
ftp_folder <- "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/900/660/285/GCA_900660285.1_BRENAR_v1/"

###### -- model management and import -----------------------------------------

config_path <- hub_download(repo_id = repo,
                            filename = "config.json")
vocab_path <- hub_download(repo_id = repo,
                           filename = "vocab.txt")
weights_path <- hub_download(repo_id = repo,
                             filename = "model.safetensors")

config <- fromJSON(txt = config_path)
weights <- safe_load_file(path = weights_path,
                          framework = "torch")

hidden_size <- config$hidden_size
n_layers <- config$num_hidden_layers
n_heads <- config$num_attention_heads
intermediate_size <- config$intermediate_size
vocab_size <- config$vocab_size
layer_norm_eps <- if (!is.null(config$layer_norm_eps)) config$layer_norm_eps else 1e-5
head_dim <- hidden_size / n_heads


# compatability check on dimensions and heads
if (head_dim != as.integer(head_dim)) {
  stop("hidden_size (",
       hidden_size,
       ") is not evenly divisible by n_heads (",
       n_heads,
       ") -- check config.json, something is wrong with these values.")
}

# initial printout for model information:
cat("model config:\n  hidden_size = ",
    hidden_size,
    "\n",
    "  n_layers = ",
    n_layers,
    "\n",
    "  n_heads = ",
    n_heads,
    "\n",
    "  head_dim = ",
    head_dim,
    "\n",
    "  intermediate_size = ",
    intermediate_size,
    "\n",
    "  vocab_size = ",
    vocab_size,
    "\n",
    sep = "")

###### -- tokenizer setup -----------------------------------------------------

# there's a final line warning here, but it should still be ingested correctly
vocab_lines <- readLines(vocab_path)
# token id == line number
# the model expects these to be zero-indexed, so we offset the seq call
# standard for models in general? or just esm?
token_to_id <- setNames(seq_along(vocab_lines) - 1L, vocab_lines)

required_specials <- c("<cls>", "<eos>", "<pad>", "<unk>")
missing_specials <- setdiff(required_specials, names(token_to_id))
if (length(missing_specials) > 0) {
  stop("vocab.txt is missing expected special token(s): ",
       paste(missing_specials,
             collapse = ", "),
       " -- check the downloaded vocab file, this tokenizer cannot proceed safely.")
}

# these aren't thematically unique to esm, but could be labelled differently
# in similar models?
cls_id <- token_to_id[["<cls>"]]
eos_id <- token_to_id[["<eos>"]]
pad_id <- token_to_id[["<pad>"]]
unk_id <- token_to_id[["<unk>"]]
mask_id <- token_to_id[["<mask>"]]

###### -- instantiate the model -----------------------------------------------

# this is an R6 constructor so it doesn't behave analogously to a
# 'normal' R functions
model <- esm_encoder(vocab_size,
                     hidden_size, n_layers,
                     n_heads,
                     intermediate_size,
                     layer_norm_eps)

copy_into(model$embed_tokens$weight,
          "esm.embeddings.word_embeddings.weight",
          weights)

# the 'layer' object instantiated in the loop copies back into
# 'model$layer[[i]]' because R6 objects are ... not strictly R-like
for (i in seq_len(n_layers)) {
  hf_i <- i - 1  # offset to zero-base
  prefix <- paste0("esm.encoder.layer.",
                   hf_i,
                   ".")
  # index to one-base
  layer <- model$layers[[i]]
  
  copy_into(layer$pre_attention_ln$weight,
            paste0(prefix, "attention.LayerNorm.weight"),
            weights)
  copy_into(layer$pre_attention_ln$bias,
            paste0(prefix, "attention.LayerNorm.bias"),
            weights)
  
  copy_into(layer$attention$query$weight,
            paste0(prefix, "attention.self.query.weight"),
            weights)
  copy_into(layer$attention$query$bias,
            paste0(prefix, "attention.self.query.bias"),
            weights)
  copy_into(layer$attention$key$weight,
            paste0(prefix, "attention.self.key.weight"),
            weights)
  copy_into(layer$attention$key$bias,
            paste0(prefix, "attention.self.key.bias"),
            weights)
  copy_into(layer$attention$value$weight,
            paste0(prefix, "attention.self.value.weight"),
            weights)
  copy_into(layer$attention$value$bias,
            paste0(prefix, "attention.self.value.bias"),
            weights)
  copy_into(layer$attention$output$weight,
            paste0(prefix, "attention.output.dense.weight"),
            weights)
  copy_into(layer$attention$output$bias,
            paste0(prefix, "attention.output.dense.bias"),
            weights)
  
  copy_into(layer$pre_ffn_ln$weight,
            paste0(prefix, "LayerNorm.weight"),
            weights)
  copy_into(layer$pre_ffn_ln$bias,
            paste0(prefix, "LayerNorm.bias"),
            weights)
  
  copy_into(layer$intermediate$weight,
            paste0(prefix, "intermediate.dense.weight"),
            weights)
  copy_into(layer$intermediate$bias,
            paste0(prefix, "intermediate.dense.bias"),
            weights)
  copy_into(layer$ffn_output$weight,
            paste0(prefix, "output.dense.weight"),
            weights)
  copy_into(layer$ffn_output$bias,
            paste0(prefix, "output.dense.bias"),
            weights)
  
  cat("loaded layer", hf_i, "\n")
}

copy_into(model$final_ln$weight,
          "esm.encoder.emb_layer_norm_after.weight",
          weights)
copy_into(model$final_ln$bias,
          "esm.encoder.emb_layer_norm_after.bias",
          weights)

# this switches from training to inference?
model$eval()

device <- if (cuda_is_available()) {
  "cuda"
} else {
  "cpu"
}
model$to(device = device)

###### -- initial success check -----------------------------------------------

test_seq <- "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGDRQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

sanity_batch <- tokenize_batch(aa_strings = test_seq,
                               token_to_id = token_to_id,
                               unk_id = unk_id)

sanity_input_ids <- sanity_batch$input_ids$to(device = device)
sanity_attention_mask <- sanity_batch$attention_mask$to(device = device)

with_no_grad({
  test_output <- model(sanity_input_ids,
                       sanity_attention_mask)
})
cat("======\n")
cat("sanity check output shape (batch, seq_len, hidden_size):",
    paste(dim(test_output),
          collapse = " x "),
    "\n")
cat("expected hidden_size:",
    hidden_size,
    "\n")
cat("======\n")

###### -- load in a proteome and get the embeddings ---------------------------

folder_split <- strsplit(x = ftp_folder,
                         split = "/",
                         fixed = TRUE)
faa_add <- paste0(ftp_folder,
                  "/",
                  folder_split[[1]][10],
                  "_protein.faa.gz")
seqs <- readAAStringSet(faa_add)

# ESM2 was trained on sequences up to roughly 1024 tokens (including <cls>/
# <eos>); anything longer is out-of-distribution for this checkpoint and is
# skipped here rather than silently truncated, which would change the protein's
# meaning. inspect `skipped_indices` afterward if this count is non-trivial.
max_tokens <- 1024
usable <- width(seqs) <= (max_tokens - 2)
skipped_indices <- which(!usable)
seqs_to_process <- seqs[usable]
batch_size <- 8
n_batches <- ceiling(length(seqs_to_process) / batch_size)
batch_map <- sort(rep(seq(n_batches),
                      length.out = length(seqs_to_process)))

# set up the search:
embeddings_list <- vector(mode = "list",
                          length = length(seqs_to_process))
names(embeddings_list) <- names(seqs_to_process)

for (d1 in seq_len(n_batches)) {
  # subset to our current batch
  w1 <- which(batch_map == d1)
  curr_strings <- as.character(seqs_to_process[w1])
  curr_len <- nchar(seqs_to_process[w1])
  curr_tokens <- tokenize_batch(aa_strings = curr_strings,
                                token_to_id = token_to_id,
                                unk_id = unk_id)
  
  # extract information...
  input_ids <- curr_tokens$input_ids
  attn_mask <- curr_tokens$attention_mask
  
  # send to our device...
  input_ids <- input_ids$to(device = device)
  attn_mask <- attn_mask$to(device = device)
  
  # generate hidden states
  with_no_grad({
    hidden_states <- model(input_ids, attn_mask)
  })
  
  # what kind of object is hidden states here? 
  for (d2 in seq_along(w1)) {
    real_len <- curr_len[d2]
    protein_embedding <- hidden_states[d2, 2:(real_len + 1), ]$to(device = "cpu")
    embeddings_list[[w1[d2]]] <- as.matrix(protein_embedding)
  }
  
  cat("processed batch",
      d1,
      "of",
      n_batches,
      "\n")
}
rm(hidden_states)

###### -- get the pseudo-likelihood scores ------------------------------------
# i.e. does this sequence look like something i've been trained on

# once again, r6 things ...
lm_head <- esm_lm_head(hidden_size,
                       vocab_size,
                       layer_norm_eps)

copy_into(lm_head$dense$weight,
          "lm_head.dense.weight",
          weights)
copy_into(lm_head$dense$bias,
          "lm_head.dense.bias",
          weights)
copy_into(lm_head$layer_norm$weight,
          "lm_head.layer_norm.weight",
          weights)
copy_into(lm_head$layer_norm$bias,
          "lm_head.layer_norm.bias",
          weights)
# copy_into(lm_head$decoder$weight,
#           "lm_head.decoder.weight",
#           weights)
copy_into(lm_head$decoder$weight,
          "esm.embeddings.word_embeddings.weight",
          weights)
copy_into(lm_head$bias,
          "lm_head.bias",
          weights)

lm_head$eval()
lm_head$to(device = device)

initial_test <- pseudo_likelihood(aa_string = as.character(seqs_to_process[1]),
                                  token_to_id = token_to_id,
                                  unk_id = unk_id)

pBar <- txtProgressBar(style = 1)
PBAR <- length(seqs_to_process)
PBAR <- 100
likelihood_vec <- vector(mode = "numeric",
                         length = PBAR)
for (d1 in seq_along(likelihood_vec)) {
  likelihood_vec[d1] <- pseudo_likelihood(aa_string = as.character(seqs_to_process[d1]),
                                          token_to_id = token_to_id,
                                          unk_id = unk_id)
  
  setTxtProgressBar(pb = pBar,
                    value = d1 / PBAR)
}
close(pBar)



