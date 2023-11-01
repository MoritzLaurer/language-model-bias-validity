library(lme4)
library(sjPlot)
library(tidyverse)
library(stringr)
library(arrow)

setwd("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/")

d = read_parquet("./results/df_test_concat.parquet.gzip") |>
  # this analysis is only conducted on experiments with biased training data
  filter(data_train_biased == TRUE) |>
  # new columns: biased_row is 1 if the text in df_test came from the biased group df_train was sampled from
  # using (partial) string matching here because group_members_train are 3 countries like "USA|DEU|FRA" and group_members_test is only one string like "USA"
  #mutate(biased_row = as.numeric(group_members_test == group_members_train), error = as.numeric(label_pred != label_gold)) |>
  mutate(biased_row = as.numeric(str_detect(group_members_train, fixed(group_members_test)))) |>
  # add error column
  mutate(error = as.numeric(label_pred != label_gold)) |>
  # clean classifier names
  mutate(classifier = recode(method, nli_short = "BERT-NLI", nli_void = "BERT-NLI-void", standard_dl = "BERT-base", "classical_ml" = "logistic reg.")) |>
  mutate(training_run = file_name) 
  

# TODO: check why:
#coronanet - continent & year raise warning with glmr: boundary (singular) fit: see help('isSingular')
#test <- d %>% filter(dataset %in% c("coronanet") & group_col %in% c("continent"))


# The ordering here decides which method is the reference method
# take BERT-NLI as reference category, since main argument in paper is about NLI.
d$classifier <- factor(as.factor(d$classifier), levels = c("BERT-NLI", "BERT-NLI-void", "BERT-base", "logistic reg."))


### single model without intercept
m_single_nointercept = glmer(error ~ -1 + classifier + classifier:biased_row + (1 | training_run), family=binomial, data=d)
tab_model(m_single_nointercept)
summary(m_single_nointercept)
plot_single_easier = plot_model(m_single_nointercept, type='pred', terms=c('classifier','biased_row'))
plot_single_easier

# Extract fixed effects coefficients
coefficients <- fixef(m_single_nointercept)
# Calculate odds ratios
odds_ratios <- exp(coefficients)
# probabilities without bias
odds_ratio_to_prob <- function(x) x / (x+1)
probability_error_without_bias <- sapply(odds_ratios[1:4],  odds_ratio_to_prob)
# probabilities with bias
probability_error_with_bias <- sapply(odds_ratios[1:4] * odds_ratios[5:8],  odds_ratio_to_prob)

bias_benefit <- probability_error_without_bias - probability_error_with_bias


### model per data + group var
models = list()
for (dataset in unique(d$dataset)) {
  for (group_col in unique(d$group_col)) {
    ds = d[d$dataset == dataset & d$group_col == group_col,]
    if (nrow(ds) == 0) next
    label = paste0(dataset, ' - ', group_col)
    message(label)
    #models[[label]] = glmer(error ~ classifier*biased_row + (1 | training_run), family=binomial, data=ds)
    models[[label]] = glmer(error ~ -1 + classifier + classifier:biased_row + (1 | training_run), family=binomial, data=ds)
    
  }
}
tab_model(models, dv.labels = names(models), show.ci=F, show.se=F, p.style='stars')

plots = list()
for (model in names(models)) {
  plots[[model]] = plot_model(models[[model]], type='pred', title=model, terms=c('classifier','biased_row'))
}
plot_grid(plots, tags=rep('', length(plots)))





### single model with intercept
# not used in paper

m_single = glmer(error ~ classifier*biased_row + (1 | training_run), family=binomial, data=d)
tab_model(m_single)
summary(m_single)
plot_single <- plot_model(m_single, type='pred', terms=c('classifier','biased_row'))
plot_single


# export model outputs for paper
library(stargazer)
stargazer(m_single, type="text")
library(broom.mixed)
tidy_output <- tidy(m_single)
glance_output <- glance(m_single)




