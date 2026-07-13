torch example
================
Nicholas Cooley
2026-07-13

``` r
library(torch)
library(hfhub)
library(safetensors)
library(Biostrings)
```

    ## Loading required package: BiocGenerics

    ## Loading required package: generics

    ## 
    ## Attaching package: 'generics'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.difftime, as.factor, as.ordered, intersect, is.element, setdiff,
    ##     setequal, union

    ## 
    ## Attaching package: 'BiocGenerics'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     IQR, mad, sd, var, xtabs

    ## The following objects are masked from 'package:base':
    ## 
    ##     anyDuplicated, aperm, append, as.data.frame, basename, cbind,
    ##     colnames, dirname, do.call, duplicated, eval, evalq, Filter, Find,
    ##     get, grep, grepl, is.unsorted, lapply, Map, mapply, match, mget,
    ##     order, paste, pmax, pmax.int, pmin, pmin.int, Position, rank,
    ##     rbind, Reduce, rownames, sapply, saveRDS, table, tapply, unique,
    ##     unsplit, which.max, which.min

    ## Loading required package: S4Vectors

    ## Loading required package: stats4

    ## 
    ## Attaching package: 'S4Vectors'

    ## The following object is masked from 'package:utils':
    ## 
    ##     findMatches

    ## The following objects are masked from 'package:base':
    ## 
    ##     expand.grid, I, unname

    ## Loading required package: IRanges

    ## Loading required package: XVector

    ## Loading required package: Seqinfo

    ## 
    ## Attaching package: 'Biostrings'

    ## The following object is masked from 'package:base':
    ## 
    ##     strsplit

``` r
library(jsonlite)

source(file = "R/esm_helpers.R")
```

``` r
# hf model
repo <- "facebook/esm2_t12_35M_UR50D"
# data to build an example from
# a genbank submission that was annotated by the submitter:
# https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_900660285.1/
ftp_folder <- "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/900/660/285/GCA_900660285.1_BRENAR_v1/"
```

``` r
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
```

``` r
# there's a final line warning here, but it should still be ingested correctly
vocab_lines <- readLines(vocab_path)
```

    ## Warning in readLines(vocab_path): incomplete final line found on
    ## '/home/n.cooley/.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots/6fbf070e65b0b7291e7bbcd451118c216cff79d8/vocab.txt'

``` r
# token id == line number
# the model expects these to be zero-indexed, so we offset the seq call
# standard for models in general? or just esm?
token_to_id <- setNames(seq_along(vocab_lines) - 1L, vocab_lines)

required_specials <- c("<cls>", "<eos>", "<pad>", "<unk>")
missing_specials <- setdiff(required_specials, names(token_to_id))

# these aren't thematically unique to esm, but could be labelled differently
# in similar models?
cls_id <- token_to_id[["<cls>"]]
eos_id <- token_to_id[["<eos>"]]
pad_id <- token_to_id[["<pad>"]]
unk_id <- token_to_id[["<unk>"]]
mask_id <- token_to_id[["<mask>"]]
```

``` r
# this is an R6 constructor so it doesn't behave analogously to
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
  
  # cat("loaded layer", hf_i, "\n")
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
```

``` r
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
# additional subset to make the example more palatable 
seqs_to_process <- seqs[usable][40]
batch_size <- 8
n_batches <- ceiling(length(seqs_to_process) / batch_size)
batch_map <- sort(rep(seq(n_batches),
                      length.out = length(seqs_to_process)))

# set up
pBar <- txtProgressBar(style = 1)
PBAR <- length(seqs_to_process)
embeddings_list <- vector(mode = "list",
                          length = PBAR)
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
  
  # cat("processed batch",
  #     d1,
  #     "of",
  #     n_batches,
  #     "\n")
  setTxtProgressBar(pb = pBar,
                    value = d1 / PBAR)
}
```

    ## ================================================================================

``` r
close(pBar)
```

``` r
rm(hidden_states)

save(embeddings_list,
     file = "embeddings.RData",
     compress = "xz")
```
