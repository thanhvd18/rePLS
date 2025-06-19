# Load required libraries
rm(list=ls())
library(circlize)
library(pracma)
library(dplyr)
library(graphics)

# Set plotting parameters
par(bg='#FFFFFF')

# Define networks and corresponding colors
networks <- c("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
colors <- c('#650871', '#97d7d7', '#235731', '#d30589', '#555533', '#e38421', '#c53a48')
bg_col <- "#FFFFFF"

# Load data
df_pvalue <- read.csv('figures/figure1/age_X_pvalue.csv')
df <- read.csv('figures/figure1/XYZ_normalize.csv')
df_barplot_strength <- read.csv('figures/figure1/barplot_data.csv')




# Save output as SVG

svg("figures/figure1/1a/circos_plot.svg", width = 8, height = 8)


# Initialize Circos Plot
circos.par(gap.degree=0, track.margin=c(0.0, 0.0), track.height=0.01)
circos.initialize(factors=networks, xlim=c(0, 8))

# Boxplot by Age Group

ylim <- c(1.75, 3.3)
circos.track(factors=networks, ylim=ylim, panel.fun=function(x, y) {
  pos_list <- linspace(CELL_META$xlim[1] + CELL_META$xrange/8, CELL_META$xlim[2] - CELL_META$xrange/8, 4)
  col_list <- c('#66c2a5', '#8da0cb', '#fc8d62', '#8856a7')
  for (i in seq(4)) {
    value <- df %>% filter(AGE_group == i - 1) %>% select(networks[CELL_META$sector.numeric.index])
    circos.boxplot(value, pos_list[i], col=col_list[i], box_width=0.5, outline=FALSE)
  }
  p <- df_pvalue$p[CELL_META$sector.numeric.index]
  nstar <- ifelse(p < 0.001, 3, ifelse(p < 0.01, 2, ifelse(p < 0.05, 1, sprintf('P=%1.2f', p))))
  circos.text(CELL_META$xcenter, CELL_META$cell.ylim[2] + convert_y(-0.25, "mm"), paste(rep('*', nstar), collapse=''),
              facing="outside", cex=2, adj=c(0.5, 0), niceFacing=TRUE)
}, bg.col=bg_col, track.height=0.3)

# Pvalue
circos.trackPlotRegion(factors=networks, track.index = get.current.track.index(),panel.fun=function(x, y) {
  circos.text(CELL_META$xcenter, CELL_META$cell.ylim[2] + convert_y(6, "mm"), CELL_META$sector.index,
              facing="bending.outside", cex=1.5, adj=c(0.5, 0), niceFacing=TRUE, col=1)
}, bg.border=FALSE, track.height=0.3)


# Violin Plot by Gender
ylim <- c(1.7, 3)
circos.trackPlotRegion(factors=networks, ylim=ylim, panel.fun=function(x, y) {
  pos_list <- linspace(CELL_META$xlim[1] + CELL_META$xrange/4, CELL_META$xlim[2] - CELL_META$xrange/4, 2)
  col_list <- c('#ff9da6', '#9ecae9')
  for (i in seq(2)) {
    value <- df %>% filter(PTGENDER == ifelse(i == 1, 1, 0)) %>% select(networks[CELL_META$sector.numeric.index])
    circos.violin(value, pos_list[i], col=col_list[i], violin_width=0.75)
  }
  male <- df %>% filter(PTGENDER == 1) %>% select(networks[CELL_META$sector.numeric.index])
  female <- df %>% filter(PTGENDER == 0) %>% select(networks[CELL_META$sector.numeric.index])
  p <- t.test(male, female)$p.value
  nstar <- ifelse(p < 0.001, 3, ifelse(p < 0.01, 2, ifelse(p < 0.05, 1, sprintf('P=%1.2f', p))))
  circos.text(CELL_META$xcenter, CELL_META$cell.ylim[2] + convert_y(-1, "mm"), paste(rep('*', nstar), collapse=''),
              facing="outside", cex=ifelse(is.numeric(nstar), 2, 1.2), adj=c(0.5, 0), niceFacing=TRUE)
}, bg.col=bg_col, track.height=0.2)


# Barplot for Network Strength
circos.track(ylim=c(-0.75, 0.95), panel.fun=function(x, y) {
  value <- data.matrix(df_barplot_strength)[, CELL_META$sector.numeric.index]
  circos.barplot(value, linspace(CELL_META$xlim[1] + CELL_META$xrange/9, CELL_META$xlim[2] - CELL_META$xrange/9, 8),
                 col=ifelse(value > 0, '#d73027', '#4575b4'), bar_width=0.2)
  circos.axis(h=0, labels.facing='inside', labels.niceFacing=TRUE, direction="inside", labels=FALSE, major.tick=FALSE)
}, track.height=0.3, bg.col=NA, bg.border=NA)


dev.off()
