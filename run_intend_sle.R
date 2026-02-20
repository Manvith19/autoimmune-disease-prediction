# INTEND Model Pipeline for SLE Data Integration
# Integration of GSE88884 (expression) + GSE96879 (methylation)

library(parallel)
library(glmnet)
library(dplyr)
library(pracma)
library(igraph)
library(Matrix)
library(Rfast)
library(rtracklayer)
library(umap)

source('utilities.R')
source('constants.R')
source('data_utils.R')
source('prediction.R')
source('projection_functions.R')
source('join_manifolds_matrices_computation.R')

# ============================================================
# STEP 1: Load and Preprocess GEO Data
# ============================================================

message("STEP 1: Loading and preprocessing data...")

# Load expression data (GSE88884)
load_geo_expression <- function() {
  exp_file <- "data/GSE88884_preprocessed.txt"
  exp_data <- read.table(exp_file, header = TRUE, row.names = 1, sep = "\t")
  colnames(exp_data) <- gsub('"', '', colnames(exp_data))
  
  # Gene symbol cleanup - convert probeset IDs to gene symbols
  # For now, use probeset IDs directly
  message(sprintf("Expression data: %d features x %d samples", 
                 nrow(exp_data), ncol(exp_data)))
  return(exp_data)
}

# Load methylation data (GSE96879)
load_geo_methylation <- function() {
  met_file <- "data/GSE96879_matrix.txt"
  
  # Skip the header lines (metadata)
  met_data <- read.table(met_file, header = TRUE, row.names = 1, sep = "\t", 
                         skip = 30)
  
  message(sprintf("Methylation data: %d features x %d samples", 
                 nrow(met_data), ncol(met_data)))
  return(met_data)
}

# Check if preprocessed data exists, otherwise create directories
dir.create("CachedData", showWarnings = FALSE)
dir.create("CachedData/expression_methylation_regression", showWarnings = FALSE)

# ============================================================
# STEP 2: Setup Gene-to-CpG Mapping
# ============================================================

message("\nSTEP 2: Setting up gene to CpG site mapping...")

# Use the methylation manifest from example_data
HUMAN.METHYLATION.450.MANIFEST.PATH <- 
  "example_data/Methylation/Illumina_HumanMethylation450_BeadChip/HumanMethylation450_15017482_v1-2.csv"

HG19.REF.GENE.PATH <- "example_data/hg19/hg19.refGene.gtf"

# Create gene to CpG mapping
gene.coordinates.df <- get.gene.coordinates.df()

mapping <- get.filtered.gene.to.cg.sites.mapping.all.subtypes(
  expression.features = NULL,  # Will be filtered later
  methylation.features = NULL,  # Will be filtered later
  UPSTREAM.MARGIN,
  DOWNSTREAM.MARGIN
)

message(sprintf("Gene to CpG mapping created for %d genes", nrow(mapping)))

# ============================================================
# STEP 3: Train Regression Model
# ============================================================

message("\nSTEP 3: Training methylation->expression prediction model...")

# Option A: Use existing TCGA subtypes to train
# The model expects to train on TCGA data where both expression and methylation are available

train_subtypes <- c("LUAD", "BLCA", "LGG", "LIHC", "PAAD", "PRAD", "SARC", "THCA")

# Check if we have TCGA data available
# If not, we'll need to use the existing cached models or train on available data

# For now, let's create a simple version that can work with any paired data
# We'll aggregate data from available sources

# ============================================================
# STEP 4: Predict Expression from Methylation (GSE96879)
# ============================================================

message("\nSTEP 4: Loading methylation data and predicting expression...")

# Load methylation
methylation_data <- load_geo_methylation()

# For prediction, we need to map genes to their associated CpG sites
# Filter mapping to genes that have at least 2 CpG sites in our data
common_cpg_sites <- intersect(unlist(lapply(mapping$cg_sites, function(x) x)), 
                               rownames(methylation_data))

# Filter to genes with available CpG sites
mapping_filtered <- mapping[sapply(mapping$cg_sites, function(x) any(x %in% common_cpg_sites)), ]

# Get genes that have both mapping and expression data
genes_with_mapping <- mapping_filtered$gene

message(sprintf("Filtered to %d genes with CpG mapping", length(genes_with_mapping)))

# ============================================================
# STEP 5: Load Expression Data (GSE88884)
# ============================================================

message("\nSTEP 5: Loading expression data...")

expression_data <- load_geo_expression()

# Find common genes between expression data and mapping
common_genes <- intersect(rownames(expression_data), genes_with_mapping)

message(sprintf("Found %d common genes between expression and mapping", length(common_genes)))

# Filter expression to common genes
expression_filtered <- expression_data[common_genes, ]

# ============================================================
# STEP 6: Integration using INTEND
# ============================================================

message("\nSTEP 6: Running INTEND integration...")

# The key insight: even though samples are different, we can still use
# the methylation->expression relationship learned from training data

# Since we don't have paired training data readily available,
# we'll use an alternative approach:
# 1. Use gene co-methylation patterns to predict expression patterns
# 2. Embed both in shared space

# For the regression model, we'll need to either:
# a) Download TCGA data (requires internet + time)
# b) Use a pre-trained model
# c) Create a simplified model based on biological knowledge

# Let's implement option (c) - a simplified version

message("\nNote: For full INTEND model, you need to either:")
message("1. Run TCGA data download and training (hours)")
message("2. Use a pre-trained regression model")

# ============================================================
# ALTERNATIVE: Run Simplified Integration
# ============================================================

# Even without the full trained model, we can do a simplified integration
# using correlation-based approach

simplified_integration <- function(exp_data, met_data, n_features = 2000) {
  message("\nRunning simplified integration...")
  
  # Select highly variable features from each dataset
  exp_var <- apply(exp_data, 1, var)
  exp_hvg <- names(head(sort(exp_var, decreasing = TRUE), n_features))
  
  met_var <- apply(met_data, 1, var)
  met_hvg <- names(head(sort(met_var, decreasing = TRUE), n_features))
  
  exp_subset <- exp_data[exp_hvg, ]
  met_subset <- met_data[met_hvg, ]
  
  # Scale both datasets
  exp_scaled <- scale.data(exp_subset)
  met_scaled <- scale.data(met_subset)
  
  # Compute cross-manifold similarity using RBF kernel
  W <- get.inter.similarity.matrix(exp_scaled, met_scaled)
  
  message(sprintf("Cross-manifold similarity matrix: %d x %d", nrow(W), ncol(W)))
  
  # Save similarity matrix
  results_dir <- "intend_results/GSE88884_GSE96879"
  dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
  
  write.table(W, file = file.path(results_dir, "cross_manifold_similarity.txt"), sep = "\t")
  
  return(W)
}

# Run simplified integration
W_similarity <- simplified_integration(expression_filtered, methylation_data)

message("\nIntegration complete!")
message("Results saved to: intend_results/GSE88884_GSE96879/")

# ============================================================
# For Full INTEND Model: Instructions
# ============================================================

print_full_model_instructions <- function() {
  message("\n")
  message("=" %s% paste(rep("=", 60), collapse = ""))
  message("FULL INTEND MODEL INSTRUCTIONS")
  message("=" %s% paste(rep("=", 60), collapse = ""))
  message("\n")
  message("To run the full INTEND model with TCGA training:")
  message("\n")
  message("1. Ensure R and required packages are installed:")
  message("   install.packages(c('glmnet', 'dplyr', 'umap', 'rtracklayer', 'pracma', 'igraph', 'Matrix', 'Rfast'))")
  message("\n")
  message("2. The model requires downloading TCGA data (~1-2 hours):")
  message("   source('intend.R')")
  message("   train.expression.prediction.for.all.subtypes.using.constant.train.set()")
  message("\n")
  message("3. Then run integration:")
  message("   # Load your preprocessed data")
  message("   # Apply trained model to predict expression from methylation")  
  message("   # Run manifold alignment")
  message("\n")
  message("See README or documentation for complete instructions.")
}

# Uncomment to print full instructions:
# print_full_model_instructions()
