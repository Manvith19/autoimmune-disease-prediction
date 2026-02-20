# INTEND Model - Data Preprocessing and Setup Script
# This script prepares the GEO data and sets up the environment

# ============================================================
# PART 1: Install Required R Packages
# ============================================================

install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Core packages required by INTEND
core_packages <- c(
  "glmnet",      # LASSO regression for expression prediction
  "dplyr",       # Data manipulation
  "parallel",    # Parallel processing
  "umap",        # UMAP dimensionality reduction
  "pracma",      # Practical math utilities
  "igraph",      # Graph operations
  "Matrix",      # Matrix operations
  "Rfast",       # Fast R functions
  "rtracklayer"  # Genome annotation import
)

install_if_missing(core_packages)

# Additional packages for manifold alignment
extra_packages <- c("mstknnclust", "dnet")
install_if_missing(extra_packages)

# ============================================================
# PART 2: Preprocess GEO Data
# ============================================================

library(limma)
library(GEOquery)

# Function to load and preprocess expression data (GSE88884)
load_expression_data <- function(data_dir = "data") {
  message("Loading GSE88884 expression data...")
  
  # Read the preprocessed expression file
  exp_file <- file.path(data_dir, "GSE88884_preprocessed.txt")
  
  if (!file.exists(exp_file)) {
    stop("Expression file not found: ", exp_file)
  }
  
  # Read expression matrix
  exp_data <- read.table(exp_file, header = TRUE, row.names = 1, sep = "\t")
  
  # Remove quotes from column names if present
  colnames(exp_data) <- gsub('"', '', colnames(exp_data))
  
  message(sprintf("Expression data loaded: %d genes x %d samples", 
                  nrow(exp_data), ncol(exp_data)))
  
  return(exp_data)
}

# Function to load and preprocess methylation data (GSE96879)
load_methylation_data <- function(data_dir = "data") {
  message("Loading GSE96879 methylation data...")
  
  # Read the series matrix to get sample information
  series_file <- file.path(data_dir, "GSE96879_series_matrix.txt")
  
  if (!file.exists(series_file)) {
    stop("Methylation series matrix not found: ", series_file)
  }
  
  # The methylation data needs to be downloaded from GEO
  # or extracted from the raw files
  message("Note: Full methylation matrix needs to be downloaded from GEO")
  message("Expected file: GSE96879_matrix.txt (beta values)")
  
  # For now, return instructions
  return(NULL)
}

# ============================================================
# PART 3: Prepare Data for INTEND Model
# ============================================================

# The INTEND model expects:
# 1. Expression matrix: genes x samples
# 2. Methylation matrix: CpG sites x samples  
# 3. Gene to CpG site mapping (from genome annotation)

# Key requirement: For TRAINING, you need PAIRED data
# (same samples with both expression and methylation measured)

# Function to prepare training data format
prepare_training_data <- function(expression_matrix, methylation_matrix, 
                                  gene_to_cpg_mapping) {
  # Filter to common genes
  common_genes <- intersect(rownames(expression_matrix), 
                           names(gene_to_cpg_mapping))
  
  exp_filtered <- expression_matrix[common_genes, ]
  
  # For each gene, get associated CpG sites
  # and filter methylation to those sites
  # ... (additional filtering logic)
  
  return(list(
    expression = exp_filtered,
    methylation = methylation_filtered,
    mapping = gene_to_cpg_mapping[common_genes]
  ))
}

# ============================================================
# PART 4: Instructions for Running with Unpaired Data
# ============================================================

print_instructions <- function() {
  message("\n")
  message("=" %s% paste(rep("=", 60), collapse = ""))
  message("IMPORTANT: The INTEND model requires PAIRED training data")
  message("=" %s% paste(rep("=", 60), collapse = ""))
  message("\n")
  message("Your datasets (GSE88884 and GSE96879) are NOT paired.")
  message("\n")
  message("Options to proceed:")
  message("1. Use TCGA data to train the model (recommended)")
  message("   - The model will download ~1000+ paired samples from TCGA")
  message("   - Run: source('intend.R'); train.expression.prediction.for.all.subtypes.using.constant.train.set()")
  message("\n")
  message("2. Find paired blood expression-methylation data")
  message("   - Search GEO for datasets with both expression + methylation")
  message("   - Look for: 'expression profiling by array' AND 'methylation profiling by array'")
  message("\n")
  message("3. If you have your own paired data:")
  message("   - Format: Expression matrix (genes x samples)")
  message("   - Format: Methylation matrix (CpG sites x samples)")
  message("   - Run training and prediction functions from prediction.R")
  message("\n")
}

# Run if executed directly
if (!interactive()) {
  print_instructions()
}
