library(tidyverse)

setwd("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/")

cisize = function(group, value) {
  groups = unique(group)
  if (length(groups) != 2) stop("There should be exactly two groups")
  x = value[group == groups[1]]
  y = value[group == groups[2]]
  if (length(x) == 1 | length(y) == 1) {
    warning(glue::glue("Not enough cases: |x|={length(x)}, |y|={length(y)}"))
    m = c(meandiff=mean(x) - mean(y), lwr.ci=NA, upr.ci=NA)
  } else {
    m = DescTools::MeanDiffCI(x, y)
  }
  as_tibble(as.list(m))
}


d = read_csv("./results/df_results.csv.gz") |> 
  mutate(method=fct_reorder(method, eval_f1_macro, .fun=mean, .desc = T),
         bias=ifelse(group_sample_strategy=="randomall", "Random", "Biased")) |>
  select(task, group, method, bias, f1_macro=eval_f1_macro) |> 
  #filter(!is.na(method)) |> 
  group_by(task, group, method) |>
  arrange(task, group, method, bias) |>
  mutate(task = recode(task, "pimpo-simple" = "PImPo", "cap-merge"="CAP-2", "cap-sotu"="CAP-SotU",
                       "coronanet"="CoronaNet")) |>
  #mutate(f1_macro_dev=f1_macro - mean(f1_macro))  |>
  ungroup()



### create disaggregated plot

d2 = d |>
  group_by(task, group, method, bias) |> 
  summarize(sd=sd(f1_macro), f1_macro=mean(f1_macro), n=n()) 

d3b <- d |> 
  group_by(task, group, method) |>
  summarize(x=cisize(fct_rev(bias), f1_macro)) |> 
  unnest_wider(x)

d4 = rbind(add_column(d2, f="F1 macro"),
           add_column(d3b, bias="diff", f="Bias penalty")) |>
     mutate(f=fct_rev(f))


plot_disaggregated <- ggplot(d4, aes(x=f1_macro, y=method, color=bias)) + 
  geom_point(data=filter(d4, bias=="diff"), aes(x=meandiff)) + 
  geom_point(data=filter(d4, bias=="Random"), position=position_nudge(y=.1)) + 
  geom_point(data=filter(d4, bias=="Biased"), position=position_nudge(y=-.1)) + 
  geom_segment(data=filter(d4, bias=="Random"), 
               position=position_nudge(y=.1),
               aes(x=f1_macro-sd, xend=f1_macro+sd, yend=method)) + 
  geom_vline(data=filter(d4, bias=="diff") |> add_column(x=0), 
             mapping=aes(xintercept=x), color="grey", lty=2) + 
  geom_segment(data=filter(d4, bias=="diff"),
               aes(x=lwr.ci, xend=upr.ci, yend=method)) + 
  geom_segment(data=filter(d4, bias=="Biased"), 
               position=position_nudge(y=-.1),
               aes(x=f1_macro-sd, xend=f1_macro+sd, yend=method)) + 
  ggh4x::facet_nested(task + group ~ f, scales = "free", space = "free") + 
  theme_classic() + 
  scale_color_discrete(name="Training data sampling strategy", breaks=c("Biased", "Random")) + 
  theme(panel.grid.major.y = element_line(),
        legend.position = "bottom") + 
  xlab("") + 
  ylab("")

plot_disaggregated
plot_disaggregated_data <- plot_disaggregated$data




### create aggregated plot
d2t = d |>
  group_by(task) |>
  mutate(f1_macro_b=f1_macro-mean(f1_macro)) |> 
  group_by(method, bias) |> 
  summarize(sd=sd(f1_macro_b), f1_macro_b=mean(f1_macro_b), f1_macro=mean(f1_macro), n=n()) 


d3t <- d |> 
  group_by(task) |>
  mutate(f1_macro=f1_macro-mean(f1_macro)) |> 
  group_by(method) |>
  summarize(x=cisize(fct_rev(bias), f1_macro)) |> 
  unnest_wider(x)

d5 = rbind(
  add_column(d2t, f="F1 macro relative to task mean", g="F1 macro averaged over all datasets and groups",
             task="A total", group="A total"),
  add_column(d3t, f="Bias penalty", g=f, bias="diff", task="A total", group="A total")
) |>
  mutate(f=fct_rev(f), g=fct_rev(g))


# Using global mean, but confidence interval based on task-centered means
plot_aggregated <- ggplot(d5, aes(x=f1_macro, y=method, color=bias)) + 
  geom_point(data=filter(d5, bias=="diff"), aes(x=meandiff)) + 
  geom_point(data=filter(d5, bias=="Random"), position=position_nudge(y=.1)) + 
  geom_point(data=filter(d5, bias=="Biased"), position=position_nudge(y=-.1)) + 
  geom_segment(data=filter(d5, bias=="Random"), 
               position=position_nudge(y=.1),
               aes(x=f1_macro-sd, xend=f1_macro+sd, yend=method)) + 
  geom_vline(data=filter(d5, bias=="diff") |> add_column(x=0), 
             mapping=aes(xintercept=x), color="grey", lty=2) + 
  geom_segment(data=filter(d5, bias=="diff"),
               aes(x=lwr.ci, xend=upr.ci, yend=method)) + 
  geom_segment(data=filter(d5, bias=="Biased"), 
               position=position_nudge(y=-.1),
               aes(x=f1_macro-sd, xend=f1_macro+sd, yend=method)) + 
  ggh4x::facet_nested(. ~ g, scales = "free", space = "free") + 
  theme_classic() + 
  scale_color_discrete(name="Training data sampling strategy", breaks=c("Biased", "Random")) +
  scale_x_continuous(breaks=c(.5 + (0:10*.05),  -.025, 0)) +
  theme(panel.grid.major.y = element_line(),
        legend.position = "bottom") + 
  xlab("") + 
  ylab("")

plot_aggregated
plot_aggregated_data <- plot_aggregated$data



# Using task-centered means
plot_aggregated_relative <- ggplot(d5, aes(x=f1_macro_b, y=method, color=bias)) + 
  geom_point(data=filter(d5, bias=="diff"), aes(x=meandiff)) + 
  geom_point(data=filter(d5, bias=="Random"), position=position_nudge(y=.1)) + 
  geom_point(data=filter(d5, bias=="Biased"), position=position_nudge(y=-.1)) + 
  geom_segment(data=filter(d5, bias=="Random"), 
               position=position_nudge(y=.1),
               aes(x=f1_macro_b-sd, xend=f1_macro_b+sd, yend=method)) + 
  geom_vline(data=filter(d5, bias=="diff") |> add_column(x=0), 
             mapping=aes(xintercept=x), color="grey", lty=2) + 
  geom_vline(xintercept=0, color="grey", lty=2) + 
  geom_segment(data=filter(d5, bias=="diff"),
               aes(x=lwr.ci, xend=upr.ci, yend=method)) + 
  geom_segment(data=filter(d5, bias=="Biased"), 
               position=position_nudge(y=-.1),
               aes(x=f1_macro_b-sd, xend=f1_macro_b+sd, yend=method)) + 
  ggh4x::facet_nested(. ~ f, scales = "free", space = "free") + 
  theme_classic() + 
  scale_color_discrete(name="Training data sampling strategy", breaks=c("Biased", "Random")) + 
  theme(panel.grid.major.y = element_line(),
        legend.position = "bottom") + 
  xlab("") + 
  ylab("")


