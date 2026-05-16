library("rstudioapi") 
setwd(dirname(getActiveDocumentContext()$path))

library(colorspace)
q20 <- qualitative_hcl(20, "Warm")

library(gganimate)

library(colleyRstats)
colleyRstats::colleyRstats_setup()

library(dplyr)
library(stringr)
library(readr)
library(ggplot2)
library(ARTool)
library(easystats)
library(readxl)
library(FSA)
library(lme4)

### GLMM additions ----------------------------------------------------------
# Loaded alongside existing libraries; do not replace anything above.
library(lmerTest)     # Satterthwaite df / p-values for lmer
library(ordinal)      # cumulative link mixed models for ordinal Likerts
library(effectsize)   # partial eta^2 from anova tables
library(ggeffects)    # marginal predictions for plotting
library(performance)  # model diagnostics
library(broom.mixed)  # tidy output
### -------------------------------------------------------------------------

main_df <- read_xlsx(path = "all_combined_prepared.xlsx")
main_df <- as.data.frame(main_df)
names(main_df)


main_df$INTRODUCTION[main_df$INTRODUCTION == "ambigious"] <- "ambiguous"

main_df$SCENARIO[main_df$SCENARIO == "3Spurig"] <- "Highway"
main_df$SCENARIO[main_df$SCENARIO == "NeueMitte"] <- "City"
main_df$SCENARIO[main_df$SCENARIO == "Spielstrasse"] <- "Walking Zone"
main_df$SCENARIO[main_df$SCENARIO == "Ueberland"] <- "Cross-country"

# Make sure the subject column is a factor
main_df$INTRODUCTION <- as.factor(main_df$INTRODUCTION)
main_df$SCENARIO <- as.factor(main_df$SCENARIO)
#main_df$ProlificID <- as.factor(main_df$ProlificID)
main_df$mIoU <- as.factor(main_df$mIoU)

# data is already prepared


levels(main_df$INTRODUCTION)
levels(main_df$SCENARIO)
levels(main_df$mIoU)

labels_xlab <- c("1" = "1")


###
# This code aims to append "_a" to the ProlificID for all but the last combination of SCENARIO and INTRODUCTION within each group of ProlificID.
# This is necessary as some participants participitated twice, thus, the art function has issues with it
main_df <- main_df |>
  group_by(ProlificID) |>
  mutate(combination = paste(SCENARIO, INTRODUCTION)) |>
  mutate(ProlificID = ifelse(combination == last(combination), 
         as.character(ProlificID), paste0(ProlificID, "_a"))) |>
  ungroup()


# because of as.character
main_df$ProlificID <- as.factor(main_df$ProlificID)

main_df <- na.omit(main_df)

# have only 7 or 12, 19, 18  entries
main_df <- subset(main_df, ProlificID %!in% c("5edabe1c28a45e161cd15325", "5c8f9fa8dfbf30001697845d", "63f79194eb27c9dc523185bd", "5e9c8deb90dd470441c7f98e"))

levels(main_df$ProlificID)

#writexl::write_xlsx(main_df, "./all_combined_prepared.xlsx")


test <- remove_outliers_REI(main_df, range = c(1,7))
test_removed <- subset(test, Suspicious != "Yes")
writexl::write_xlsx(test_removed, "./all_combined_prepared_removed_REI.xlsx")


### GLMM additions: continuous mIoU + helper functions ----------------------
# Continuous mIoU (factor version is preserved above for the ART code).
main_df$mIoU_num <- as.numeric(as.character(main_df$mIoU))
# Standardised version: makes the intercept the mean-mIoU condition and
# improves convergence of random-slope models.
main_df$mIoU_c <- as.numeric(scale(main_df$mIoU_num, center = TRUE, scale = TRUE))

# Sum-to-zero contrasts for the categorical factors so type-III anova tests
# main effects against grand-mean (not against an arbitrary reference cell).
contrasts(main_df$INTRODUCTION) <- contr.sum(nlevels(main_df$INTRODUCTION))
contrasts(main_df$SCENARIO)     <- contr.sum(nlevels(main_df$SCENARIO))

# ---- LMM helper -----------------------------------------------------------
# Fits the full INTRODUCTION * SCENARIO * mIoU_c model with a random slope
# on mIoU_c by ProlificID. If that's singular or fails to converge, falls
# back to a random-intercept model. Reports Satterthwaite anova + partial
# eta squared with 95% CIs.
run_lmm <- function(dv, data = main_df,
                    fixed = "INTRODUCTION * SCENARIO * mIoU_c",
                    random_full   = "(1 + mIoU_c | ProlificID)",
                    random_simple = "(1 | ProlificID)",
                    verbose = TRUE) {
  if (verbose) cat("\n=== LMM:", dv, "===\n")
  f_full   <- as.formula(paste(dv, "~", fixed, "+", random_full))
  f_simple <- as.formula(paste(dv, "~", fixed, "+", random_simple))

  m <- tryCatch(
    lmerTest::lmer(f_full, data = data, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa",
                                         optCtrl = list(maxfun = 2e5))),
    error = function(e) NULL,
    warning = function(w) suppressWarnings(
      lmerTest::lmer(f_full, data = data, REML = TRUE,
                     control = lmerControl(optimizer = "bobyqa",
                                           optCtrl = list(maxfun = 2e5)))
    )
  )
  used_random <- "random_slope"
  if (is.null(m) || lme4::isSingular(m, tol = 1e-4)) {
    if (verbose) cat("(random-slope singular/failed; using random-intercept)\n")
    m <- lmerTest::lmer(f_simple, data = data, REML = TRUE,
                        control = lmerControl(optimizer = "bobyqa"))
    used_random <- "random_intercept"
  }

  a  <- anova(m, type = 3, ddf = "Satterthwaite")
  es <- tryCatch(
    effectsize::eta_squared(a, partial = TRUE, ci = 0.95,
                            alternative = "two.sided"),
    error = function(e) NULL
  )
  if (verbose) {
    print(a)
    if (!is.null(es)) print(es)
  }
  invisible(list(model = m, anova = a, eta2 = es, random = used_random))
}

# ---- CLMM helper (ordinal Likert) -----------------------------------------
# For 5- or 7-point Likert items, the LMM treats integer ratings as
# continuous. As a robustness check, fit a cumulative link mixed model
# (proportional odds) with the same fixed structure.
run_clmm <- function(dv, data = main_df,
                     fixed  = "INTRODUCTION * SCENARIO * mIoU_c",
                     random = "(1 | ProlificID)",
                     verbose = TRUE) {
  if (verbose) cat("\n=== CLMM (ordinal):", dv, "===\n")
  d <- data
  d[[dv]] <- ordered(d[[dv]])
  f <- as.formula(paste(dv, "~", fixed, "+", random))
  m <- tryCatch(
    ordinal::clmm(f, data = d, Hess = TRUE),
    error = function(e) { message("CLMM failed: ", conditionMessage(e)); NULL }
  )
  if (!is.null(m) && verbose) {
    print(summary(m))
    # Wald type-II tests on the fixed effects (no Anova method for clmm in
    # base ordinal; using a refit clm + car::Anova would ignore the random
    # effect, so we report Wald z from the coef table above).
  }
  invisible(m)
}

# ---- Plot helper: marginal predictions from an LMM ------------------------
# Drop-in alternative to the stat_summary plots that uses model-implied
# means + 95% CIs rather than raw cell means.
plot_lmm <- function(fit, terms = c("mIoU_c [all]", "INTRODUCTION", "SCENARIO"),
                     ylab = "Predicted") {
  pred <- ggeffects::ggpredict(fit$model, terms = terms)
  ggplot2::ggplot(pred, ggplot2::aes(x = x, y = predicted,
                                     colour = group, fill = group,
                                     group = group)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = conf.low, ymax = conf.high),
                         alpha = 0.15, colour = NA) +
    ggplot2::geom_line(linewidth = 1.2) +
    ggplot2::facet_wrap(~ facet) +
    ggplot2::labs(x = "mIoU (z-scored)", y = ylab,
                  colour = NULL, fill = NULL) +
    easystats::theme_lucid()
}
### -------------------------------------------------------------------------



#### CLean Data ####

# this would remove 10 data points

# remove_group_outliers <- function(original_data, group, group_column = "mIoU") {
#   # Subset the data for the current group based on the configurable group_column
#   group_data <- subset(original_data, original_data[[group_column]] == group)
#   
#   # Identify outliers using the performance::check_outliers function
#   outliers_list <- performance::check_outliers(group_data[["trust"]])
#   
#   # Create a dataframe containing only the outliers
#   filtered_data <- group_data[as.vector(outliers_list), ]
#   
#   # Remove the outliers from the original data set for the current group
#   cleaned_data <- anti_join(original_data, filtered_data, by = c("ProlificID", "INTRODUCTION", "NETWORK"))
#   
#   return(cleaned_data)
# }
# 
# 
# 
# 
# # Get unique levels of 'group_id'
# unique_groups <- unique(main_df$INTRODUCTION)
# 
# for(group in unique_groups) {
#   
#   main_df <- remove_group_outliers(main_df, group = group, group_column = "INTRODUCTION")
# }
# 
# unique_groups <- unique(main_df$SCENARIO)
# 
# 
# for(group in unique_groups) {
#   
#   main_df <- remove_group_outliers(main_df, group = group, group_column = "SCENARIO")
# }
# 
# unique_groups <- unique(main_df$mIoU)
# 
# 
# for(group in unique_groups) {
#   
#   main_df <- remove_group_outliers(main_df, group = group, group_column = "mIoU")
# }





#main_df |> group_by(ProlificID) |> count() |> print(n=500)
main_df$tr

main_df |> group_by(mIoU) |> summarise(mean = mean(trust), sd = sd(trust))|> writexl::write_xlsx("./test.xlsx")


########### TLX
checkAssumptionsForAnova(data = main_df, y = "TLX1", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(TLX1 ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "mental workload")

### GLMM ###
fit_TLX1 <- run_lmm("TLX1")

d <- FSA::dunnTest(TLX1 ~ mIoU, data = main_df, method = "holm")
d
#reportDunnTest(main_df = main_df, d = d, iv = "mIoU", dv = "TLX1")



main_df |> ggplot() +
  aes(x = SCENARIO, y = TLX1, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Mental workload") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5) # 95 % mean_cl_boot is 95% confidence intervals
#ggsave("plots/ps_score_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



########### Trust
checkAssumptionsForAnova(data = main_df, y = "predictability", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(predictability ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "predictability")

### GLMM ###
fit_predictability <- run_lmm("predictability")
# 5-point Likert → also check ordinal model:
clmm_predictability <- run_clmm("predictability")

d <- dunnTest(predictability ~ SCENARIO, data = main_df, method = "holm")
d
reportDunnTest(main_df = main_df, d = d, iv = "SCENARIO", dv = "predictability")
reportDunnTestTable(main_df = main_df, iv = "SCENARIO", dv = "predictability")



main_df |> ggplot() +
  labs(subtitle = "Range: 1 - 5")+
  aes(x = SCENARIO, y = predictability, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
  theme_lucid(axis.text.size = 25) +
  scale_color_see() + 
  ylab("Predictability") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/predictability.png", width = 12, height = 9,)

checkAssumptionsForAnova(data = main_df, y = "trust", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))

modelArt <- art(trust ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "trust")

### GLMM ###
fit_trust <- run_lmm("trust")
clmm_trust <- run_clmm("trust")

main_df$mIoUNUmber <- as.numeric(as.character(main_df$mIoU))

lmer(trust ~  mIoU + (1 | ProlificID), data = main_df) |> easystats::model_dashboard(output_file = "trust_lm.html")
modelTrust <- lm(trust ~  mIoUNUmber, data = main_df)
modelTrustlmer <- lmer(trust ~  mIoUNUmber + (1 | ProlificID), data = main_df)

new_data_lmer <- data.frame(mIoU = seq(1, 100, by = 1))
new_data_lmer$ProlificID <- 1
#new_data_lmer$mIoU <- as.factor(new_data_lmer$mIoU)




library(merTools)
#preds <- merTools::predictInterval(modelTrustlmer, newdata = new_data_lmer)
pred <- cbind(new_data_lmer, predictInterval(modelTrustlmer, new_data_lmer))


# Create ggplot
pred |> ggplot() + 
  geom_line(aes(x = mIoU, y = fit)) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  geom_ribbon(aes(x = mIoU, ymin = lwr, ymax = upr), alpha = .2) +
  geom_point(data = main_df, aes(x = mIoUNUmber, y = trust), color = 'cyan', alpha = 0.1, size = 3) +
  ylab("Trust")



# LM without the random slope
# Generate new data for prediction
new_data <- data.frame(mIoU = seq(1, 100, by = 1))

new_data$predlm = predict(modelTrust, newdata = new_data, interval = "confidence")
#new_data$predlm[,1]

ggplot(new_data, aes(x = mIoU, y = predlm[,1]) ) +
  geom_point() +
  geom_ribbon( aes(ymin = predlm[,2], ymax = predlm[,3]), alpha = .15) +
  geom_line( aes(y = predlm[,1]), size = 1) +
  geom_point(data = main_df, aes(x = mIoUNUmber, y = trust), color = 'cyan', alpha = 0.1, size = 3) +




dunnTest(trust ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "trust")


my_animation <- main_df |> ggplot() +
  labs(subtitle = "Range: 1 - 5")+
  aes(x = mIoU, y = trust, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
  theme_lucid(axis.text.size = 25) +
  scale_color_see() + 
  ylab("Trust") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)+
  #Create animation
  transition_states(mIoU, transition_length = 2, state_length = 1) +
  enter_fade() +
  shadow_mark() 

gganimate::animate(my_animation, width = 1920, height = 1080, renderer = av_renderer("plots/trust_mIoU.mp4"))

#gganimate::anim_save("plots/trust_mIoU.mp4")

main_df |> ggplot() +
  labs(subtitle = "Range: 1 - 5")+
  aes(x = mIoU, y = trust, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
  theme_lucid(axis.text.size = 15, axis.text.angle = 45) +
  scale_color_see() + 
  ylab("Trust") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.25, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/trust.pdf", width = 12, height = 9, device = cairo_pdf)
ggsave("plots/trust.png", width = 12, height = 9,)




main_df |> ggplot() +
  labs(subtitle = "Range: 1 - 5")+
  aes(x = mIoU, y = trust) +
  theme_lucid(axis.text.size = 25) +
  scale_color_see() + 
  ylab("Trust") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)



main_df |> ggplot() +
  labs(subtitle = "Range: 1 - 5")+
  aes(x = SCENARIO, y = trust, fill = mIoU, colour = mIoU, group = mIoU) +
  theme_lucid(axis.text.size = 25) +
  scale_color_see() + 
  ylab("Trust") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "none", legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.2) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.15)
  #stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)




###### SART ######




checkAssumptionsForAnova(data = main_df, y = "SA", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))

modelArt <- art(SA ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "situation awareness")

### GLMM ###
fit_SA <- run_lmm("SA")

d <- dunnTest(SA ~ mIoU, data = main_df, method = "holm")
d
reportDunnTest(main_df = main_df, d = d, iv = "mIoU", dv = "SA")

main_df |> ggplot() +
  aes(x = SCENARIO, y = SA, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
  theme_lucid(axis.text.size = 25) +
  scale_color_see() + 
  ylab("Situation awareness") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = c(0.85, 0.25), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/sa_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


# Demand
checkAssumptionsForAnova(data = main_df, y = "Demand", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(Demand ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "Demand")
#The ART found a significant main effect of \mIoU on Demand (\F{19}{2147}{2.50}, \pminor{0.001}). 

### GLMM ###
fit_Demand <- run_lmm("Demand")

dunnTest(Demand ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "Demand")


main_df |> ggplot() +
  aes(x = mIoU, y = Demand, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
  theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Demand") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

# Supply
checkAssumptionsForAnova(data = main_df, y = "Supply", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(Supply ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "Supply")
#The ART found no significant effects on Supply.

### GLMM ###
fit_Supply <- run_lmm("Supply")

main_df |> ggplot() +
  aes(x = mIoU, y = Supply, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Supply") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)


# Understanding
checkAssumptionsForAnova(data = main_df, y = "Understanding", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(Understanding ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "Understanding")
# The ART found a significant main effect of \mIoU on Understanding (\F{19}{2147}{1.76}, \p{0.022}). 

### GLMM ###
fit_Understanding <- run_lmm("Understanding")

dunnTest(Understanding ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "Understanding")


main_df |> ggplot() +
  aes(x = mIoU, y = Understanding, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Understanding") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)








###### Perceived Safety ######
checkAssumptionsForAnova(data = main_df, y = "perceivedSafety_score", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))

modelArt <- art(perceivedSafety_score ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "Perceived Safety")

### GLMM ###
fit_perceivedSafety <- run_lmm("perceivedSafety_score")

main_df |> ggplot() +
  aes(x = SCENARIO, y = perceivedSafety_score, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Perceived Safety") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)




library(randomcoloR)
n <- 20
palette <- distinctColorPalette(n)


main_df |> ggplot() +
  aes(x = INTRODUCTION, y = perceivedSafety_score, fill = mIoU, colour = mIoU, group = mIoU) +
  scale_colour_manual(values = palette) +
  ylab("Perceived Safety") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = c(0.88, 0.5), legend.text = element_text(size = myfontsize - 24)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.2) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.2) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.2)
ggsave("plots/ps_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



reportARTCombine <- function(model, dv = "Testdependentvariable", write_to_clipboard = FALSE) {
  # Check that the model and dependent variable are not empty
  assertthat::not_empty(model)
  assertthat::not_empty(dv)
  
  # Initialize an empty list to store main effects
  main_effects <- list()
  
  # Check if the model has a "Pr(>F)" column
  if ("Pr(>F)" %!in% colnames(model)) {
    cat(paste0("No column ``Pr(>F)'' was found."))
  } else {
    # Check if any p-values are significant
    if (!any(model$`Pr(>F)` < 0.05, na.rm = TRUE)) {
      message_to_write <- paste0("The ART found no significant effects on ", dv, ". ")
      if (write_to_clipboard) {
        write_clip(message_to_write)
      } else {
        cat(message_to_write)
      }
    } else {
      model$descriptions <- model[,1]
      model$descriptions <- gsub(":", " X", model$descriptions)
      
      for (i in 1:length(model$`Pr(>F)`)) {
        if (!is.na(model$`Pr(>F)`[i]) && model$`Pr(>F)`[i] < 0.05) {
          Fvalue <- round(model$`F value`[i], digits = 2)
          numeratordf <- model$Df[i]
          denominatordf <- model$Df.res[i]
          pValueNumeric <- model$`Pr(>F)`[i]
          pValue <- if (pValueNumeric < 0.001) "\\pminor{0.001}" else sprintf("\\p{%.3f}", round(pValueNumeric, digits = 3))
          
          effect_type <- if (str_detect(model$descriptions[i], "X")) "interaction" else "main"
          
          # Store main effects in the list
          if (effect_type == "main") {
            main_effects <- append(main_effects, list(list(effect_name = model$descriptions[i], Fvalue = Fvalue, numeratordf = numeratordf, denominatordf = denominatordf, pValue = pValue)))
          }
          
          stringtowrite <- paste0("The ART found a significant ", effect_type, " effect of ", trimws(model$descriptions[i]), " on ", dv, " (\\F{", numeratordf, "}{", denominatordf, "}{", sprintf("%.2f", Fvalue), "}, ", pValue, "). ")
          stringtowrite <- gsub("(?<=\\s)X", "$\\\\times$ \\\\", stringtowrite, perl = TRUE)
          
          if (write_to_clipboard) {
            write_clip(stringtowrite)
          } else {
            cat(stringtowrite)
          }
        }
      }
      
      # If there are exactly two main effects, combine them into one sentence
      if (length(main_effects) == 2) {
        effect1 <- main_effects[[1]]
        effect2 <- main_effects[[2]]
        combined_string <- paste0("The ART found significant main effects of ", effect1$effect_name, " (\\F{", effect1$numeratordf, "}{", effect1$denominatordf, "}{", sprintf("%.2f", effect1$Fvalue), "}, ", effect1$pValue, ") and ", effect2$effect_name, " (\\F{", effect2$numeratordf, "}{", effect2$denominatordf, "}{", sprintf("%.2f", effect2$Fvalue), "}, ", effect2$pValue, ") on ", dv, ".")
        
        if (write_to_clipboard) {
          write_clip(combined_string)
        } else {
          cat(combined_string)
        }
      }
    }
  }
}




###### Custom SUS ######

checkAssumptionsForAnova(data = main_df, y = "freq", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(freq ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "use frequently")
# The ART found a significant main effect of \mIoU on use frequently (\F{19}{2147}{6.86}, \pminor{0.001})

### GLMM ###
fit_freq <- run_lmm("freq")
clmm_freq <- run_clmm("freq")

d <- dunnTest(freq ~ mIoU, data = main_df, method = "holm")
d
dunnTest(freq ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "freq")

main_df |> ggplot() +
  aes(x = mIoU, y = freq, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Use frequently") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)


checkAssumptionsForAnova(data = main_df, y = "complex", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(complex ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "unnecessarily complex")
#The ART found a significant main effect of \mIoU on unnecessarily complex (\F{19}{2147}{4.30}, \pminor{0.001}).

### GLMM ###
fit_complex <- run_lmm("complex")
clmm_complex <- run_clmm("complex")

dunnTest(complex ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "complex")
#A post-hoc test found that 67.87 was significantly higher (\m{2.47}, \sd{1.47}) in terms of \complex compared to 86.2 (\m{1.81}, \sd{1.14}; \padj{0.043}). 

main_df |> ggplot() +
  aes(x = mIoU, y = complex, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("unnecessarily complex") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

###### Perception of User ######

#Recognized Pedestrians
checkAssumptionsForAnova(data = main_df, y = "p1b", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(p1b ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of recognized pedestrians")
#The ART found a significant main effect of \SCENARIO on assessment of recognized pedestrians (\F{3}{113}{3.06}, \p{0.031}). The ART found a significant main effect of \mIoU on assessment of recognized pedestrians (\F{19}{2147}{5.43}, \pminor{0.001}).

### GLMM ###
fit_p1b <- run_lmm("p1b")
clmm_p1b <- run_clmm("p1b")

dunnTest(p1b ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "p1b")
dunnTest(p1b ~ SCENARIO, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "SCENARIO", dv = "p1b")
dunnTest(p1b ~ SCENARIO, data = main_df, method = "holm") |> reportDunnTestTable(main_df = main_df, iv = "SCENARIO", dv = "p1b")


main_df |> ggplot() +
  aes(x = mIoU, y = p1b, fill = SCENARIO, colour = SCENARIO, group = SCENARIO) +
    theme_lucid(axis.text.size = 25, axis.text.angle = 45) +   scale_color_see() + 
  ylab("Recognized pedestrians") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = c(0.8, 0.15), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.5) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/recog_pedestrians.png", width = 12, height = 9,)


#Recognized Vehicles
checkAssumptionsForAnova(data = main_df, y = "p2b", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))

modelArt <- art(p2b ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of recognized vehicles")
# The ART found a significant main effect of \mIoU on assessment of recognized vehicles (\F{19}{2147}{6.88}, \pminor{0.001}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \mIoU on assessment of recognized vehicles (\F{19}{2147}{1.71}, \p{0.029}).

### GLMM ###
fit_p2b <- run_lmm("p2b")
clmm_p2b <- run_clmm("p2b")

dunnTest(p2b ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "p2b")

main_df |> ggplot() +
  aes(x = mIoU, y = p2b, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25, axis.text.angle = 45) +   scale_color_see() + 
  ylab("Recognized vehicles") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = c(0.85, 0.25), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/rec_vehicle_interaction.pdf", width = 12, height = 9, device = cairo_pdf)
ggsave("plots/recog_vehicle_interaction.png", width = 12, height = 9,)



#Recognized Signposts
checkAssumptionsForAnova(data = main_df, y = "p3b", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(p3b ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of recognized signposts")
# The ART found a significant main effect of \mIoU on assessment of recognized signposts (\F{19}{2147}{5.28}, \pminor{0.001})

### GLMM ###
fit_p3b <- run_lmm("p3b")
clmm_p3b <- run_clmm("p3b")

dunnTest(p3b ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "p3b")


main_df |> ggplot() +
  aes(x = mIoU, y = p3b, fill = SCENARIO, colour = SCENARIO, group = SCENARIO) +
    theme_lucid(axis.text.size = 25, axis.text.angle = 45) +   scale_color_see() + 
  ylab("Recognized signposts") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/recog_signposts.png", width = 12, height = 9,)




#Predict pedestrian intentions
checkAssumptionsForAnova(data = main_df, y = "pintention", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(pintention ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of predicted pedestrian intention")
# The ART found a significant main effect of \mIoU on assessment of predicted pedestrian intention (\F{19}{2147}{3.85}, \pminor{0.001}).

### GLMM ###
fit_pintention <- run_lmm("pintention")
clmm_pintention <- run_clmm("pintention")

dunnTest(pintention ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "pintention")


main_df |> ggplot() +
  
  aes(x = mIoU, y = pintention, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Predicted pedestrian intention") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#Predict vehicle paths
checkAssumptionsForAnova(data = main_df, y = "ppaths", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(ppaths ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of predicted vehicle paths")
#The ART found a significant main effect of \mIoU on assessment of predicted vehicle paths (\F{19}{2147}{3.14}, \pminor{0.001}). 

### GLMM ###
fit_ppaths <- run_lmm("ppaths")
clmm_ppaths <- run_clmm("ppaths")

dunnTest(ppaths ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "ppaths")


main_df |> ggplot() +
  
  aes(x = mIoU, y = ppaths, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("predicted vehicle paths") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#longitudinal control
checkAssumptionsForAnova(data = main_df, y = "p4b", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(p4b ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "longitudinal control")
#The ART found a significant main effect of \mIoU on longitudinal control (\F{19}{2147}{2.48}, \pminor{0.001}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO $\times$ \mIoU on longitudinal control (\F{57}{2147}{1.38}, \p{0.034}).

### GLMM ###
fit_p4b <- run_lmm("p4b")
clmm_p4b <- run_clmm("p4b")

dunnTest(p4b ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "p4b")


main_df |> ggplot() +
  
  aes(x = mIoU, y = p4b, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Longitudinal control") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#lateral control
checkAssumptionsForAnova(data = main_df, y = "p5b", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(p5b ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "lateral control")
# The ART found no significant effects on lateral control.

### GLMM ###
fit_p5b <- run_lmm("p5b")
clmm_p5b <- run_clmm("p5b")

dunnTest(p5b ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "p5b")



main_df |> ggplot() +
  
  aes(x = mIoU, y = p5b, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("lateral control") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#unsafe Judgement
checkAssumptionsForAnova(data = main_df, y = "unsafeJudgement", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(unsafeJudgement ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of AV's Judgement")
#The ART found a significant main effect of \mIoU on assessment of AV's Judgement (\F{19}{2147}{2.23}, \p{0.002}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \mIoU on assessment of AV's Judgement (\F{19}{2147}{1.81}, \p{0.017}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO $\times$ \mIoU on assessment of AV's Judgement (\F{57}{2147}{1.83}, \pminor{0.001}). 

### GLMM ###
fit_unsafeJudgement <- run_lmm("unsafeJudgement")
clmm_unsafeJudgement <- run_clmm("unsafeJudgement")

dunnTest(unsafeJudgement ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "unsafeJudgement")


main_df |> ggplot() +
  
  aes(x = mIoU, y = unsafeJudgement, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("AV's Judgement") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/unsafeJudgement_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



p <- main_df |> ggplot() +
  
  aes(x = as.numeric(mIoU), y = unsafeJudgement, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("AV's Judgement") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  scale_x_continuous(breaks = c(min(as.numeric(main_df$mIoU)), max(as.numeric(main_df$mIoU))), labels = c("low", "high")) + 
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
p + facet_grid(~SCENARIO)
ggsave("plots/unsafeJudgement_three_way_interaction.pdf", width = 12, height = 9, device = cairo_pdf)

# , labeller = labeller(SCENARIO =
#   c(
#     "no maneuver" = "No ego trajectory",
#     "with maneuver" = "with ego trajectory"
#   )
# )

# p <- main_df |> ggplot() +
#   
#   aes(x = SCENARIO, y = unsafeJudgement, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
#     theme_lucid(axis.text.size = 25) +   scale_color_see() + 
#   ylab("AV's Judgement") +
#   theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
#   xlab("") +
#   stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
#   stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
#   stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
# p + facet_grid(~mIoU)



#react appropriately
checkAssumptionsForAnova(data = main_df, y = "reactapprop", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(reactapprop ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv ="assessment of AV's Reaction")
#The ART found a significant main effect of \mIoU on assessment of AV's Reaction (\F{19}{2147}{2.96}, \pminor{0.001})

### GLMM ###
fit_reactapprop <- run_lmm("reactapprop")
clmm_reactapprop <- run_clmm("reactapprop")

dunnTest(reactapprop ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "reactapprop")



main_df |> ggplot() +
  
  aes(x = mIoU, y = reactapprop, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("AV's Reaction") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#I would perform better
checkAssumptionsForAnova(data = main_df, y = "performbetter", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(performbetter ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of Performance")
#The ART found no significant effects on assessment of Performance.

### GLMM ###
fit_performbetter <- run_lmm("performbetter")
clmm_performbetter <- run_clmm("performbetter")


main_df |> ggplot() +
  
  aes(x = mIoU, y = performbetter, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("AV's Performance") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#Clear what the AV will do
checkAssumptionsForAnova(data = main_df, y = "willdo", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(willdo ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "clarity of next AV action")
# The ART found a significant main effect of \mIoU on clarity of next AV action (\F{19}{2147}{1.74}, \p{0.025}). 

### GLMM ###
fit_willdo <- run_lmm("willdo")
clmm_willdo <- run_clmm("willdo")

dunnTest(willdo ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "willdo")


main_df |> ggplot() +
  
  aes(x = mIoU, y = willdo, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("clarity of next AV action") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

###### Driving Style ######

checkAssumptionsForAnova(data = main_df, y = "drivingstyle", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(drivingstyle ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "assessment of driving style")
# The ART found a significant main effect of \mIoU on assessment of driving style (\F{19}{2147}{2.37}, \pminor{0.001}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \mIoU on assessment of driving style (\F{19}{2147}{1.79}, \p{0.019}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO $\times$ \mIoU on assessment of driving style (\F{57}{2147}{1.36}, \p{0.039}). 

### GLMM ###
fit_drivingstyle <- run_lmm("drivingstyle")
clmm_drivingstyle <- run_clmm("drivingstyle")

dunnTest(drivingstyle ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "drivingstyle")


main_df |> ggplot() +
  
  aes(x = as.numeric(mIoU), y = drivingstyle, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Driving style") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5) +
  scale_x_continuous(breaks = c(min(as.numeric(main_df$mIoU)), max(as.numeric(main_df$mIoU))), labels = c("low", "high"))
ggsave("plots/drivingstyle_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



p <- main_df |> ggplot() +
  
  aes(x = as.numeric(mIoU), y = drivingstyle, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Driving style") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  scale_x_continuous(breaks = c(min(as.numeric(main_df$mIoU)), max(as.numeric(main_df$mIoU))), labels = c("low", "high")) + 
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
p + facet_grid(~SCENARIO)
ggsave("plots/drivingstyle_three_way_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


###### Own Questions ######

#Drove as Expected

checkAssumptionsForAnova(data = main_df, y = "expected", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(expected ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "AV behavior conformity")
# The ART found a significant main effect of \mIoU on AV behavior conformity (\F{19}{2147}{3.43}, \pminor{0.001}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO $\times$ \mIoU on AV behavior conformity (\F{57}{2147}{1.45}, \p{0.016}). 

### GLMM ###
fit_expected <- run_lmm("expected")
clmm_expected <- run_clmm("expected")

dunnTest(expected ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "expected")


p <- main_df |> ggplot() +
  
  aes(x = as.numeric(mIoU), y = expected, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("AV behavior conformity") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  scale_x_continuous(breaks = c(min(as.numeric(main_df$mIoU)), max(as.numeric(main_df$mIoU))), labels = c("low", "high")) + 
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
p + facet_grid(~SCENARIO)
ggsave("plots/expected_three_way_interaction.pdf", width = 12, height = 9, device = cairo_pdf)


#Reasons for behavior were clear

checkAssumptionsForAnova(data = main_df, y = "clearReason", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(clearReason ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "clarity of reasons")
# The ART found a significant main effect of \mIoU on clarity of reasons (\F{19}{2147}{3.07}, \pminor{0.001}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \mIoU on clarity of reasons (\F{19}{2147}{1.84}, \p{0.015}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO $\times$ \mIoU on clarity of reasons (\F{57}{2147}{1.46}, \p{0.015}). 

### GLMM ###
fit_clearReason <- run_lmm("clearReason")
clmm_clearReason <- run_clmm("clearReason")

dunnTest(clearReason ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "clearReason")


main_df |> ggplot() +
  
  aes(x = mIoU, y = clearReason, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Clarity of reasons") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/clearReason_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



p <- main_df |> ggplot() +
  
  aes(x = as.numeric(mIoU), y = clearReason, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Clarity of reasons") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = c(0.85, 0.25), legend.text = element_text(size = 8)) +
  xlab("") +
  scale_x_continuous(breaks = c(min(as.numeric(main_df$mIoU)), max(as.numeric(main_df$mIoU))), labels = c("low", "high")) + 
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
p + facet_grid(~SCENARIO)
ggsave("plots/clearReason_three_way_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



#Visualizations were reasonable

checkAssumptionsForAnova(data = main_df, y = "reason", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(reason ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "reasonability")
# The ART found a significant main effect of \mIoU on reasonability (\F{19}{2147}{7.11}, \pminor{0.001}). 

### GLMM ###
fit_reason <- run_lmm("reason")
clmm_reason <- run_clmm("reason")

dunnTest(reason ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "reason")


main_df |> ggplot() +
  
  aes(x = mIoU, y = reason, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("reasonability") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#Visualizations were necessary

checkAssumptionsForAnova(data = main_df, y = "necc", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(necc ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "necessity")
# The ART found a significant main effect of \mIoU on necessity (\F{19}{2147}{4.44}, \pminor{0.001}). 

### GLMM ###
fit_necc <- run_lmm("necc")
clmm_necc <- run_clmm("necc")

dunnTest(necc ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "necc")


main_df |> ggplot() +
  
  aes(x = mIoU, y = necc, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("necessity") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#Too many visualizations

checkAssumptionsForAnova(data = main_df, y = "clutter", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(clutter ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "visualization complexity")
# The ART found a significant main effect of \mIoU on visualization complexity (\F{19}{2147}{4.73}, \pminor{0.001}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO on visualization complexity (\F{3}{113}{2.74}, \p{0.046}). The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO $\times$ \mIoU on visualization complexity (\F{57}{2147}{1.40}, \p{0.028}). 

### GLMM ###
fit_clutter <- run_lmm("clutter")
clmm_clutter <- run_clmm("clutter")

#lmer(clutter ~  mIoU + (1 | ProlificID), data = main_df) |> easystats::model_dashboard(output_file = "test.html")

dunnTest(clutter ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "clutter")
# A post-hoc test found that 70.23 was significantly higher (\m{3.80}, \sd{2.35}) in terms of \clutter compared to 83.68 (\m{2.76}, \sd{2.01}; \padj{0.039}). A post-hoc test found that 70.23 was significantly higher (\m{3.80}, \sd{2.35}) in terms of \clutter compared to 86.2 (\m{2.60}, \sd{1.89}; \padj{0.008}). 

main_df |> ggplot() +
  
  aes(x = SCENARIO, y = clutter, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Visualization complexity") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/clutter_interaction.pdf", width = 12, height = 9, device = cairo_pdf)



p <- main_df |> ggplot() +
  
  aes(x = as.numeric(mIoU), y = clutter, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Visualization complexity") +
  theme(legend.title = element_blank(), axis.title = element_text(size = 20), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = c(0.85, 0.9), legend.text = element_text(size = 8)) +
  xlab("") +
  scale_x_continuous(breaks = c(min(as.numeric(main_df$mIoU)), max(as.numeric(main_df$mIoU))), labels = c("low", "high")) + 
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
p + facet_grid(~SCENARIO)
ggsave("plots/clutter_three_way_interaction.pdf", width = 12, height = 9, device = cairo_pdf)




#More perception visualizations 

checkAssumptionsForAnova(data = main_df, y = "moreper", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(moreper ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "need of more perception-related information")
# The ART found no significant effects on need of more perception-related information.

### GLMM ###
fit_moreper <- run_lmm("moreper")
clmm_moreper <- run_clmm("moreper")

#dunnTest(moreper ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "moreper")


main_df |> ggplot() +
  
  aes(x = mIoU, y = moreper, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Need of more perception-related information") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#More prediction visualizations

checkAssumptionsForAnova(data = main_df, y = "morepre", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(morepre ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "need of more prediction-related information")
# The ART found no significant effects on need of more prediction-related information.

### GLMM ###
fit_morepre <- run_lmm("morepre")
clmm_morepre <- run_clmm("morepre")

#dunnTest(morepre ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "morepre")



main_df |> ggplot() +
  
  aes(x = mIoU, y = morepre, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Need of more prediction-related information") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)

#More future path visualizations

checkAssumptionsForAnova(data = main_df, y = "moreego", factors = c("INTRODUCTION", "SCENARIO", "mIoU"))


modelArt <- art(moreego ~ INTRODUCTION * SCENARIO * mIoU + Error(ProlificID / mIoU), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "need of more maneuver-related information")
# The ART found a significant interaction effect of \INTRODUCTION $\times$ \SCENARIO on need of more maneuver-related information (\F{3}{113}{3.29}, \p{0.023}).

### GLMM ###
fit_moreego <- run_lmm("moreego")
clmm_moreego <- run_clmm("moreego")

#dunnTest(moreego ~ mIoU, data = main_df, method = "holm") |> reportDunnTest(main_df = main_df, iv = "mIoU", dv = "moreego")



main_df |> ggplot() +
  
  aes(x = SCENARIO, y = moreego, fill = INTRODUCTION, colour = INTRODUCTION, group = INTRODUCTION) +
    theme_lucid(axis.text.size = 25) +   scale_color_see() + 
  ylab("Need of more maneuver-related information") +
  theme(legend.title = element_blank(), axis.title=element_text(size=22), axis.text=element_text(size=18), plot.title = element_text(size=28), plot.subtitle = element_text(size=18), legend.background = element_blank(), legend.position = "inside",  legend.position.inside = c(0.85, 0.85), legend.text = element_text(size = 8)) +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.8) + 
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", width = .5, position = position_dodge(width = 0.05), alpha = 0.5)
ggsave("plots/moreego_interaction.pdf", width = 12, height = 9, device = cairo_pdf)

###### Accumulated Scores ######







# TODO: Baseline-Vergleiche



### GLMM additions: summary table across all fitted models ------------------
# Collect every fit_* model into one tidy table for the supplement / paper.
all_fits <- ls(pattern = "^fit_")
glmm_summary <- do.call(rbind, lapply(all_fits, function(nm) {
  obj <- get(nm)
  if (is.null(obj$anova)) return(NULL)
  a  <- as.data.frame(obj$anova)
  a$term <- rownames(a)
  a$dv   <- sub("^fit_", "", nm)
  a$random_structure <- obj$random
  if (!is.null(obj$eta2)) {
    es <- as.data.frame(obj$eta2)
    a$eta2_partial <- es$Eta2_partial[match(a$term, es$Parameter)]
    a$eta2_CI_low  <- es$CI_low[match(a$term, es$Parameter)]
    a$eta2_CI_high <- es$CI_high[match(a$term, es$Parameter)]
  }
  a
}))
glmm_summary <- glmm_summary[, c("dv", "term", "NumDF", "DenDF", "F value",
                                  "Pr(>F)", "eta2_partial",
                                  "eta2_CI_low", "eta2_CI_high",
                                  "random_structure")]
# Tag effect type for filtering / sorting.
glmm_summary$effect_type <- ifelse(
  grepl(":", glmm_summary$term),
  ifelse(lengths(regmatches(glmm_summary$term, gregexpr(":", glmm_summary$term))) >= 2,
         "three-way", "two-way"),
  "main"
)
# Holm-correct p-values WITHIN dv (across the 7 anova rows).
glmm_summary <- glmm_summary |>
  dplyr::group_by(dv) |>
  dplyr::mutate(p_holm_within_dv = p.adjust(`Pr(>F)`, method = "holm")) |>
  dplyr::ungroup() |>
  as.data.frame()

writexl::write_xlsx(glmm_summary, "./glmm_summary.xlsx")
cat("\nWrote glmm_summary.xlsx with", nrow(glmm_summary), "rows across",
    length(unique(glmm_summary$dv)), "dependent variables.\n")
### -------------------------------------------------------------------------
