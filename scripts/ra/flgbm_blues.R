# RA §2.3: per-environment BLUEs as in Fernandes et al. 2024 src/blues.R, with lme4 in place of asreml
# (fixed: Hybrid + Replicate; random: Replicate:Block, Range, Pass; the same removals when a factor is absent or
# single-level). BLUE of a hybrid = intercept + hybrid effect + mean replicate effect (asreml predict, classify
# Hybrid). LOCAL_CHECK excluded as in the original.
#   Rscript flgbm_blues.R <trait_csv> <years comma list> <out_csv>
suppressMessages(library(lme4))
a <- commandArgs(TRUE)
years <- as.integer(strsplit(a[2], ",")[[1]])
d <- read.csv(a[1], stringsAsFactors = FALSE)
d <- d[d$Year %in% years & d$Hybrid != "LOCAL_CHECK" & !is.na(d$Yield_Mg_ha), ]
res <- list()
for (env in sort(unique(d$Env))) {
  x <- d[d$Env == env, ]
  for (v in c("Hybrid", "Replicate", "Block", "Range", "Pass")) x[[v]] <- factor(x[[v]])
  rnd <- c("(1|Replicate:Block)", "(1|Range)", "(1|Pass)")
  if (all(is.na(x$Range)) || nlevels(droplevels(x$Range)) < 2) rnd <- setdiff(rnd, "(1|Range)")
  if (all(is.na(x$Pass)) || nlevels(droplevels(x$Pass)) < 2) rnd <- setdiff(rnd, "(1|Pass)")
  if (nlevels(droplevels(x$Block)) < 2) rnd[rnd == "(1|Replicate:Block)"] <- "(1|Replicate)"
  fixed <- if (nlevels(droplevels(x$Replicate)) > 1) "Yield_Mg_ha ~ Hybrid + Replicate" else "Yield_Mg_ha ~ Hybrid"
  if (grepl("\\(1\\|Replicate\\)", paste(rnd, collapse = "")) && grepl("Replicate", fixed)) rnd <- setdiff(rnd, "(1|Replicate)")
  hm <- tapply(x$Yield_Mg_ha, x$Hybrid, mean)
  val <- tryCatch({
    if (length(rnd)) {
      m <- lmer(as.formula(paste(fixed, "+", paste(rnd, collapse = "+"))), data = x, REML = TRUE)
      b <- fixef(m)
    } else {
      m <- lm(as.formula(fixed), data = x); b <- coef(m)
    }
    b[is.na(b)] <- 0
    hy <- levels(droplevels(x$Hybrid))
    rep_eff <- if (grepl("Replicate", fixed)) mean(c(0, b[grep("^Replicate", names(b))])) else 0
    he <- sapply(hy, function(h) { k <- paste0("Hybrid", h); if (k %in% names(b)) b[[k]] else 0 })
    setNames(b[["(Intercept)"]] + he + rep_eff, hy)
  }, error = function(e) { message(env, ": ", conditionMessage(e), " -> plot means"); hm })
  res[[env]] <- data.frame(Env = env, Hybrid = names(val), predicted.value = as.numeric(val))
}
write.csv(do.call(rbind, res), a[3], row.names = FALSE)
cat("BLUES_DONE", length(res), "environments\n")
