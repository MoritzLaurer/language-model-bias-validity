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


# sample for faster testing
#set.seed(42)
# Get stratified sample
#d <- d %>%
#  group_by(training_run) %>%
#  sample_n(100) %>%
#  ungroup()  # Remove grouping


#m1 = glmer(error ~ -1 + classifier*biased_row + (1 | training_run), family=binomial, data=d)
#pred <- ggeffects::ggpredict(m1, terms = c("classifier", "biased_row")) |>
#  as_tibble() |>
#  mutate(intest=if_else(group==0, "Yes", "No"))


get_plotdata = function(dataset) {
  # model with intercept to get confidence intervals
  m1 = glmer(error ~ -1 + classifier*biased_row + (1 | training_run), family=binomial, data=dataset)
  pred <- ggeffects::ggpredict(m1, terms = c("classifier", "biased_row")) |>
    as_tibble() |>
    mutate(intest=if_else(group==1, "Yes", "No"))
  # model without intercept for more interpretable outputs
  m2 = glmer(error ~ -1 + classifier + classifier:biased_row + (1 | training_run), family=binomial, data=dataset)
  
  summary(m2)$coefficients |>
    as.data.frame() |> 
    rownames_to_column("param") |> 
    as_tibble() |>
    filter(str_detect(param, ":biased_row")) |>
    mutate(x=str_remove_all(param, "^classifier|:biased_row$"),
           predicted=exp(Estimate),
           conf.low=exp(Estimate-`Std. Error`), conf.high=exp(Estimate+`Std. Error`)) |>
    select(x, predicted, conf.low, conf.high) |>
    add_column(intest="Odds ratio") |>
    bind_rows(pred) |>
    select(-std.error, -group) |>
    mutate(z=if_else(intest=='Odds ratio', 'Odds ratio of error on\ngroup member seen during training\nvs. unseen group members', 'Probability of making an error ') |> fct_rev(),
           x=fct_reorder(x, predicted))
}


p_tot = get_plotdata(d)

# give explicit order to legend
p_tot$intest <- factor(p_tot$intest, levels = c("No", "Yes", "Odds ratio"))


## aggregate plot
ggplot(p_tot, aes(y=x, yend=x, x=predicted, xend=conf.high, color=intest)) + 
  geom_point(data=filter(p_tot, intest=="Yes"), position=position_nudge(y=.1)) + 
  geom_point(data=filter(p_tot, intest=="No"), position=position_nudge(y=-.1)) + 
  geom_point(data=filter(p_tot, intest=="Odds ratio")) + 
  geom_segment(data=filter(p_tot, intest=="Yes"), aes(x=conf.low), position=position_nudge(y=.1)) + 
  geom_segment(data=filter(p_tot, intest=="No"), aes(x=conf.low), position=position_nudge(y=-.1)) + 
  geom_segment(data=filter(p_tot, intest=="Odds ratio"), aes(x=conf.low)) + 
  geom_vline(data=filter(p_tot, intest=="Odds ratio") |> add_column(xx=1), 
             mapping=aes(xintercept=xx), color="grey", lty=2) + 
  theme_classic() + 
  #scale_color_discrete(name="Test text from same group member as training texts?", breaks=c("No", "Yes", "Odds ratio")) + 
  scale_color_manual(name="Test text from same group member as training texts?", 
                     values=c("Yes"="#F8766D", "No"="#619CFF", "Odds ratio"="#00BA38"),
                     breaks=c("Yes", "No", "Odds ratio")) +
  theme(panel.grid.major.y = element_line(),
        legend.position = "bottom") + 
  xlab("") + 
  ylab("") +   
  facet_grid(cols=vars(z), scales = "free", space="free")






## disaggregated analysis
d <- d |> mutate(label=str_c(dataset, group_col, sep="-"))

get_plotdata_subset <- function(lbl) {
  dataset = dataset = d |> filter(label == lbl)
  get_plotdata(dataset) |> add_column(label=lbl)
}

dsets = map(unique(d$label), get_plotdata_subset, .progress = TRUE) |> list_rbind()
dsets <- dsets |> mutate(dataset=str_extract(label, "(.*)-", group=1),
                         group=str_extract(label, ".*-(.*)", group=1))

# disaggregated plot
plot_disaggregated <- ggplot(dsets, aes(y=x, yend=x, x=predicted, xend=conf.high, color=intest)) + 
  geom_point(data=filter(dsets, intest=="Yes"), position=position_nudge(y=.1)) + 
  geom_point(data=filter(dsets, intest=="No"), position=position_nudge(y=-.1)) + 
  geom_point(data=filter(dsets, intest=="Odds ratio")) + 
  geom_segment(data=filter(dsets, intest=="Yes"), aes(x=conf.low), position=position_nudge(y=.1)) + 
  geom_segment(data=filter(dsets, intest=="No"), aes(x=conf.low), position=position_nudge(y=-.1)) + 
  geom_segment(data=filter(dsets, intest=="Odds ratio"), aes(x=conf.low)) + 
  geom_vline(data=filter(dsets, intest=="Odds ratio") |> add_column(xx=1), 
             mapping=aes(xintercept=xx), color="grey", lty=2) + 
  theme_classic() + 
  scale_color_discrete(name="Test data point from group member in training set?") + 
  theme(panel.grid.major.y = element_line(),
        legend.position = "bottom") + 
  xlab("") + 
  ylab("") +   
  ggh4x::facet_nested(dataset+group ~ z, scales = "free", space="free")


# adjust x-axis for odds ratio plot
# convert to ggplotGrob and inspect
plot_disaggregated_g <- ggplotGrob(plot_disaggregated)
plot_disaggregated_g$widths
# identify the two panels. They should have a 'null' dimension as they're relative. g$widths[5] was the width of the first panel, so let's increase it
plot_disaggregated_g$widths[5] <- unit(2, 'null')
# plot
grid::grid.draw(plot_disaggregated_g)
