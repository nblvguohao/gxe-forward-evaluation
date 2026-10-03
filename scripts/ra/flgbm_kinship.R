# RA §2.3: additive kinship as in Fernandes et al. 2024 src/kinship.R (AGHmatrix::Gmatrix, VanRaden) on the 2,425
# benchmark markers (0/1/2; the original pruned the 2022 VCF with vcftools/plink first). Missing calls -> marker mean.
#   Rscript flgbm_kinship.R <markers.parquet> <out.parquet>
suppressMessages({library(AGHmatrix); library(arrow)})
a <- commandArgs(TRUE)
g <- as.data.frame(read_parquet(a[1]))
M <- as.matrix(g[, -1]); rownames(M) <- g$genotype
M <- apply(M, 2, function(v) { v[is.na(v)] <- round(mean(v, na.rm = TRUE)); v })
maf <- colMeans(M) / 2; M <- M[, pmin(maf, 1 - maf) >= 0.01]
G <- Gmatrix(M, method = "VanRaden", missingValue = NA, maf = 0.01)
out <- data.frame(Hybrid = rownames(G), G, check.names = FALSE)
write_parquet(out, a[2])
cat("KINSHIP_DONE", dim(G), "\n")
