# CLAC decomposition (docs/prereg_clac_decomposition_2026-10-02.md). Run inside ra_decomp/Y<Y>/, which holds read-only
# links to the intermediate files of the existing CLAC run (ra/Y<Y>/). Writes decomp_<id>.csv (Env, Hybrid, pred) for
# A_full and the five ablations D1-D5. Only within-environment ranking matters, so environment means and the
# environment-SD scaling of Model A are omitted (constant within an environment).
suppressMessages(require(bWGR))
set.seed(123)
cores <- 2
load('ES_dta_QC.RData')     # ES_dta (YLD = outlier-filtered yield), PS_dta
load('AdjYld.RData')        # AdjYld: list env -> spatially adjusted hybrid means (over rownames(K))
load('K2X.RData')           # X (arc-cosine kernel reparametrised)
load('ES_MVGBLUP1.RData')   # fit_mv, Y
load('AccESPS.RData')       # A: expected accuracy, training env x target env
PS <- read.csv(list.files('Testing_Data', pattern = '^1_Submission_Template', full.names = TRUE)[1])

si_matrix <- function(Acols, equal = FALSE) {
  ES_state <- substr(rownames(Acols), 1, 2); PS_state <- substr(colnames(Acols), 1, 2)
  ES_code <- substr(rownames(Acols), 1, 4);  PS_code <- substr(colnames(Acols), 1, 4)
  SI <- Acols
  for (i in seq_len(ncol(SI))) {
    idx <- if (equal) rep(1, nrow(SI)) else
      (ES_code == PS_code[i]) * Acols[, i] * 2 + (ES_state == PS_state[i]) * Acols[, i] * 2 + Acols[, i] * 0.1
    SI[, i] <- idx / sum(idx)
  }
  SI
}
gpred <- function(b, Xm, equal = FALSE) {
  H <- Xm %*% b; colnames(H) <- colnames(Y); H <- apply(H, 2, scale)
  Acols <- A[colnames(H), , drop = FALSE]
  G <- H %*% si_matrix(Acols, equal); rownames(G) <- rownames(Xm); G
}
write_pred <- function(G, id) {
  ok <- PS$Hybrid %in% rownames(G) & PS$Env %in% colnames(G)
  out <- data.frame(Env = PS$Env[ok], Hybrid = PS$Hybrid[ok],
                    pred = G[cbind(match(PS$Hybrid[ok], rownames(G)), match(PS$Env[ok], colnames(G)))])
  write.csv(out, paste0('decomp_', id, '.csv'), row.names = FALSE)
  cat(id, 'rows', nrow(out), 'missing', sum(!ok), '\n')
}
mv_fit <- function(Ym, Xm) {
  f <- MRR3(Ym, Xm, InnerGS = TRUE, NoInv = TRUE, df0 = 10, cores = cores, tol = 1e-04)
  colnames(f$b) <- colnames(Ym); f$b
}
env_means <- function(col) {
  envs <- colnames(Y)
  M <- sapply(envs, function(e) { d <- ES_dta[ES_dta$Env == e, ]; tapply(d[[col]], d$Hybrid, mean, na.rm = TRUE)[rownames(X)] })
  rownames(M) <- rownames(X); colnames(M) <- envs; M[is.nan(M)] <- NA; M
}
t0 <- Sys.time()
# A_full and D1 reuse the fitted multivariate GBLUP
write_pred(gpred(fit_mv$b, X), 'A_full')
write_pred(gpred(fit_mv$b, X, equal = TRUE), 'D1_equal_weights')
# D2: outlier-filtered raw means (no spatial adjustment)
write_pred(gpred(mv_fit(env_means('YLD'), X), X), 'D2_no_spatial')
# D3: raw means (no outlier filter, no spatial adjustment)
write_pred(gpred(mv_fit(env_means('Yield_Mg_ha'), X), X), 'D3_no_cleaning')
# D4: linear kernel on the same QC'd markers, same adjusted phenotypes
load('gen.RData')           # gen2: hybrids x markers (0/1/2)
Z <- scale(gen2[rownames(X), ], center = TRUE, scale = FALSE); Z[is.na(Z)] <- 0
Kl <- tcrossprod(Z); Kl <- Kl / mean(diag(Kl))
E <- EigenEVD(Kl, cores = cores); w <- which(E$D > 0.1)
Xl <- E$U[, w] %*% diag(sqrt(E$D[w])); rownames(Xl) <- rownames(X)
write_pred(gpred(mv_fit(Y, Xl), Xl), 'D4_linear_kernel')
# D5: single main effect on the mean of within-environment-centred adjusted phenotypes
Yc <- sweep(Y, 2, colMeans(Y, na.rm = TRUE)); yb <- rowMeans(Yc, na.rm = TRUE); obs <- !is.na(yb)
f5 <- emRR(yb[obs], X[obs, ]); g5 <- c(X %*% f5$b)
G5 <- matrix(g5, nrow = nrow(X), ncol = ncol(A), dimnames = list(rownames(X), colnames(A)))
write_pred(G5, 'D5_single_main_effect')
cat('DECOMP_DONE', format(Sys.time() - t0), '\n')
