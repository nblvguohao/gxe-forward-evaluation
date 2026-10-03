# RA §2.2: Lopez-Cruz et al. 2023 model "SNP+EC+SNPxEC" (MAIZE-HUB analysis.zip: 1_get_G_matrix_PC.R,
# 1_get_WWt_matrix_PC.R, 1_get_model_components.R, 5_LYO_CV_model.R), re-run for a true forward target year Y.
# Same construction as the original: G = XX'/mean(diag) on centred markers, G_PC = all PCs; WWt from scaled ECs
# (HI30_ removed) over the environments in the data, W_PC = all PCs; SNPxEC = tensor EVD of G and WWt at alpha;
# BGLR BRR for the three terms, nIter 12000, burnIn 4000. Deviations (prereg §2.2): official G2F APSIM ECs; the
# 2,425 markers of the benchmark; regions pooled; alpha by the pre-registered memory rule.
#
#   Rscript run_lc.R <Y> <ec_mode: real|hist> <mode: feas|fit> <alpha>
suppressMessages({library(BGLR); library(tensorEVD); library(arrow)})
args <- commandArgs(TRUE)
Y <- as.integer(args[1]); ec_mode <- args[2]; mode <- args[3]; alpha <- as.numeric(args[4])
RA <- "<workstation>/dart-gxe/scratch_ideas_2026-09-29/ra"
out <- file.path(RA, "lc"); dir.create(out, showWarnings = FALSE)
tag <- sprintf("Y%d_%s_a%s", Y, ec_mode, format(alpha))
t0 <- Sys.time()

cells <- as.data.frame(read_parquet(file.path(RA, "lc_inputs", "cells.parquet")))       # env, year, genotype, y, scored
geno <- as.data.frame(read_parquet(file.path(RA, "lc_inputs", "markers.parquet")))      # genotype + marker columns
ec <- read.csv(file.path(RA, "lc_inputs", "ec.csv"), row.names = 1, check.names = FALSE)
ec <- ec[, -grep("^HI30_", colnames(ec)), drop = FALSE]

trn <- cells[cells$year < Y, ]
tst <- cells[cells$year == Y & cells$scored, ]
stopifnot(max(trn$year) < Y)
tst$y <- NA_real_                                   # no phenotype of year Y enters anything below
pheno <- rbind(trn, tst)
pheno <- pheno[pheno$env %in% rownames(ec) & pheno$genotype %in% geno$genotype, ]
ntrn <- sum(pheno$year < Y)

# ECs of the target environments: observed ('real') or the location mean over training environments ('hist')
E <- ec[unique(pheno$env), , drop = FALSE]
if (ec_mode == "hist") {
  loc <- function(e) sub("^([A-Z]{2}[HS][0-9]).*$", "\\1", sub("_[0-9]{4}$", "", e))
  tr_envs <- unique(trn$env[trn$env %in% rownames(ec)])
  for (e in unique(tst$env[tst$env %in% rownames(E)])) {
    same <- tr_envs[loc(tr_envs) == loc(e)]
    src <- if (length(same)) same else tr_envs
    E[e, ] <- colMeans(ec[src, , drop = FALSE], na.rm = TRUE)
  }
}
E <- E[, apply(E, 2, function(v) all(is.finite(v)) && sd(v) > 0), drop = FALSE]
WWt <- tcrossprod(scale(as.matrix(E))); WWt <- WWt / mean(diag(WWt))
W_EVD <- eigen(WWt, symmetric = TRUE); iw <- which(W_EVD$values > 1e-8)
W_PC <- sweep(W_EVD$vectors[, iw], 2, sqrt(W_EVD$values[iw]), "*"); rownames(W_PC) <- rownames(WWt)

ids <- sort(unique(pheno$genotype))
X <- as.matrix(geno[match(ids, geno$genotype), -1]); rownames(X) <- ids
X <- apply(X, 2, function(v) { v[is.na(v)] <- mean(v, na.rm = TRUE); v })
X <- scale(X, center = TRUE, scale = FALSE); X <- X[, apply(X, 2, function(v) any(v != 0)), drop = FALSE]
G <- tcrossprod(X); G <- G / mean(diag(G))
G_EVD <- eigen(G, symmetric = TRUE); ig <- which(G_EVD$values > 1e-8)
G_PC <- sweep(G_EVD$vectors[, ig], 2, sqrt(G_EVD$values[ig]), "*"); rownames(G_PC) <- ids

pheno$genotype <- factor(pheno$genotype, levels = ids)
pheno$env <- factor(pheno$env, levels = rownames(WWt))
te <- tensorEVD(G, WWt, pheno$genotype, pheno$env, alpha = alpha)
te$vectors <- sweep(te$vectors, 2, sqrt(te$values), "*")      # memory: one copy only (same values as before)
SNPxEC <- te$vectors; te$vectors <- NULL; invisible(gc())
info <- list(target = Y, ec_mode = ec_mode, alpha = alpha, n_rows = nrow(pheno), n_train = ntrn, n_test = nrow(pheno) - ntrn,
             n_gpc = ncol(G_PC), n_wpc = ncol(W_PC), n_gxe = ncol(SNPxEC), n_markers = ncol(X),
             gb_gxe = as.numeric(object.size(SNPxEC)) / 2^30, seconds_setup = as.numeric(difftime(Sys.time(), t0, units = "secs")))
print(unlist(info))
wr <- function(x, f) write.csv(as.data.frame(lapply(x, function(v) paste(v, collapse = ";"))), f, row.names = FALSE)
if (mode == "feas") { wr(info, file.path(out, paste0("feas_", tag, ".csv"))); quit(save = "no") }

SNP <- G_PC[as.character(pheno$genotype), ]; EC <- W_PC[as.character(pheno$env), ]
itr <- which(pheno$year < Y); its <- which(pheno$year == Y)
GXte <- SNPxEC[its, , drop = FALSE]; GXtr <- SNPxEC[itr, , drop = FALSE]; rm(SNPxEC); invisible(gc())
ETA <- list(SNP = list(X = SNP[itr, ], model = "BRR", saveEffects = FALSE),
            EC = list(X = EC[itr, ], model = "BRR", saveEffects = FALSE),
            SNPxEC = list(X = GXtr, model = "BRR", saveEffects = FALSE))
rm(GXtr); invisible(gc())
set.seed(20260930)
tmpd <- file.path(out, paste0("tmp_", tag, "_"));
fm <- BGLR(y = pheno$y[itr], ETA = ETA, nIter = 12000, burnIn = 4000, saveAt = tmpd, verbose = FALSE)
pred <- data.frame(env = as.character(pheno$env[its]), genotype = as.character(pheno$genotype[its]),
                   snp = as.vector(SNP[its, ] %*% fm$ETA$SNP$b), ec = as.vector(EC[its, ] %*% fm$ETA$EC$b),
                   gxe = as.vector(GXte %*% fm$ETA$SNPxEC$b))
pred$pred <- fm$mu + pred$snp + pred$ec + pred$gxe
write_parquet(pred, file.path(out, paste0("pred_", tag, ".parquet")))
unlink(Sys.glob(paste0(tmpd, "*.dat")))
info$seconds_total <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
info$varE <- fm$varE; info$varB <- c(SNP = fm$ETA$SNP$varB, EC = fm$ETA$EC$varB, SNPxEC = fm$ETA$SNPxEC$varB)
wr(info, file.path(out, paste0("fit_", tag, ".csv")))
cat("LC_DONE", tag, "\n")
