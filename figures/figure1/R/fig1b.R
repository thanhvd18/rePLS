
{ rm(list=ls())
# install.packages("circlize")
# install.packages("pracma")
library(circlize)
library(pracma)
library(dplyr)
library(graphics)
par(bg='#FFFFFF')
}
# dev.new()

outcomes <- c('CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate','RAVLT_learning', 'RAVLT_perc_forgetting')
colors <- c('#650871', '#97d7d7', '#235731', '#d30589', '#555533', '#e38421', '#c53a48', '#000000')

bg_col <- "#FFFFFF"
# 0: female, 1 male

df_pvalue <- read.csv('figures/figure1/age_Y_pvalue.csv')
df <- read.csv('figures/figure1/XYZ_normalize.csv')

cor_mat <- cor(df[outcomes])
cor_mat[abs(cor_mat) < 0.7] <- 0
cor_mat <- abs(cor_mat)


# svg("figures/figure1/1b/circos_plot_YZ.svg", width = 8, height = 8)

# config circle
circos.par(gap.degree=0, track.margin=c(0.0, 0.0), track.height=0.01)
circos.initialize( factors=outcomes,xlim=c(0,10))


# 
# # # Plot Chord Diagram
# chordDiagramFromMatrix(cor_mat, order=outcomes, symmetric=TRUE, col=1, h.ratio=0.35, scale=TRUE, annotationTrack=NULL,
#                        preAllocateTracks=2, big.gap=10, link.visible=cor_mat > 0.6)





age_list = a = df %>%
  filter( PTGENDER == 0) %>%
  select(outcomes)

ylim= c(-2.5,2.6)
circos.track(factors =outcomes,ylim = ylim, panel.fun = function(x, y) {
  pos_list = linspace(CELL_META$xlim[1]+CELL_META$xrange/8,CELL_META$xlim[2]-CELL_META$xrange/8,4)
  col_list = c('#66c2a5','#8da0cb', '#fc8d62','#8856a7')
  #4groups: 50-60,60-70,70-80,80-90
  for(i in seq(4)) {
    value =  df %>%
      filter( AGE_group == i-1) %>%
      select(outcomes[CELL_META$sector.numeric.index])
    circos.boxplot(value, pos_list[i],col=col_list[i],box_width=0.4,outline = FALSE)
  }
  p = df_pvalue$p[CELL_META$sector.numeric.index]
  if (p < 0.1/100){
    nstar =  3  
  }else if (p < 1/100){
    nstar = 2
  }else if (p < 5/100){
    nstar = 1
  }else{
    nstar = sprintf('P=%1.2f',p)
  }
  circos.text(CELL_META$xcenter, CELL_META$cell.ylim[2] + convert_y(-0.25, "mm"), 
              paste(if (class(nstar) == "numeric") rep('*',nstar) else nstar,collapse = ''),
              facing = "outside", cex = 2,
              adj = c(0.5, 0), niceFacing = TRUE)
},bg.col=bg_col,track.height=0.3) #grey90





circos.trackPlotRegion(factors=outcomes, track.index = get.current.track.index(), panel.fun = function(x, y) {
  circos.text(CELL_META$xcenter, CELL_META$cell.ylim[2] + convert_y(6, "mm"),
              paste0(CELL_META$sector.index),
              facing = "bending.outside", cex = 1.5,
              adj = c(0.5, 0), niceFacing = TRUE,col=1)

}, bg.border=F,track.height=0.3)



# ================================== violinplot
ylim = c(-3,6)
circos.trackPlotRegion(factors=outcomes,ylim=ylim, panel.fun = function(x, y) {
  value =  matrix(rnorm(100*2),ncol=2)#replicate(runif(2), n = 2, simplify = FALSE)
  # circos.violin(value, linspace(CELL_META$xlim[1]+CELL_META$xrange/4,CELL_META$xlim[2]-CELL_META$xrange/4,2), col = c('#ff9da6','#9ecae9'),cex=1,violin_width=0.35)
  # circos.violin(value, linspace(CELL_META$xlim[1]+CELL_META$xrange/4,CELL_META$xlim[2]-CELL_META$xrange/4,2), col = c('#ff9da6','#9ecae9'),cex=1,violin_width=0.75)
  pos_list = linspace(CELL_META$xlim[1]+CELL_META$xrange/4,CELL_META$xlim[2]-CELL_META$xrange/4,2)
  col_list = c('#ff9da6','#9ecae9')
  #4groups: 50-60,60-70,70-80,80-90
  for(i in seq(2)) {
    # value = gender_list[i]
    value =  df %>%
      filter( PTGENDER == (if (i ==1) 1 else 0))%>%
      select(outcomes[CELL_META$sector.numeric.index])
    circos.violin(value, pos_list[i],col=col_list[i],violin_width=0.75)
  }




  # ================ Pvalue
  male = df %>%
    filter( PTGENDER == 1 )%>%
    select(outcomes[CELL_META$sector.numeric.index])
  female = df %>%
    filter( PTGENDER == 0 )%>%
    select(outcomes[CELL_META$sector.numeric.index])
  test = t.test(male,female)
  p = test$p.value
  if (p < 0.1/100){
    nstar =  3
  }else if (p < 1/100){
    nstar = 2
  }else if (p < 5/100){
    nstar = 1
  }else{
    nstar = sprintf('P=%1.2f',p)
  }
  # ================ Pvalue
  circos.text(CELL_META$xcenter, CELL_META$cell.ylim[2] + convert_y(-1, "mm"),
              paste(if (class(nstar) == "numeric") rep('*',nstar) else nstar,collapse = ''),
              facing = "outside", cex = if (class(nstar) == "numeric") 2 else 1.2,
              adj = c(0.5, 0), niceFacing = TRUE)
},bg.col='#FFFFFF',track.height = 0.2)


dev.off()
