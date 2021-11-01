Sentiment analysis
================
Koray Poyraz
10/29/2021

# Assignment Text Mining

## Step 1: Load this data set and split it into training (80%) and test (20%) sets.

``` r
# apply regex to remove html tags
movie_df = movie_review %>% 
  mutate(review=str_replace_all(review, "<[^>]*>", ""))

## 80% of the sample size
sample_size = floor(0.80 * nrow(movie_df))

## generate samples
train_index = sample(seq_len(nrow(movie_df)), size = sample_size)

# init train and test sets
train_df = movie_df[train_index, ]
test_df = movie_df[-train_index, ]

dim(train_df)
```

    ## [1] 4000    3

``` r
dim(test_df)
```

    ## [1] 1000    3

## Step 2: Create a TFiDF representation + run a Naïve Bayes or SVM classifier

create tf idf vectors for train set

``` r
# compute in parallel
n_cores = detectCores() - 1
cl <- makePSOCKcluster(ifelse(n_cores < 6, n_cores, 6))
registerDoParallel(cl)

start_time <- Sys.time()

# select top 500 features
tfv = TfIdfVectorizer$new(
  max_features=500,
  remove_stopwords = F
  )

# create tf idf vectors for train set
train_tf_features = tfv$fit_transform(train_df$review)

# create tf idf vectors for test set
test_tf_features = tfv$transform(test_df$review)

print(str_c('Finished in: ',Sys.time() - start_time))
```

    ## [1] "Finished in: 7.93329591751099"

``` r
stopCluster(cl)
```

dims of tf idf vectors of train and test set

``` r
dim(train_tf_features)
```

    ## [1] 4000  500

``` r
dim(test_tf_features)
```

    ## [1] 1000  500

## Step 3: Run your model and get its AUC and Loss values on the test set.

prepare train control

``` r
x_train = as.data.frame(cbind(train_tf_features, sentiment = train_df$sentiment))
x_test = as.data.frame(cbind(test_tf_features, sentiment = test_df$sentiment))

# prepare train control with 10-fold and repeat 3 times
train_control = trainControl(
  method="repeatedcv",
  number=10,
  repeats=1,
  classProbs = T,
  summaryFunction = twoClassSummary
  )
```

train SVM model

``` r
x_train = x_train %>% mutate(sentiment= ifelse(sentiment == 1, 'p', 'n'))
x_test = x_test %>% mutate(sentiment= ifelse(sentiment == 1, 'p', 'n'))

# compute in parallel
n_cores = detectCores() - 1
cl <- makePSOCKcluster(ifelse(n_cores < 6, n_cores, 6))
registerDoParallel(cl)

start_time <- Sys.time()

# SVM
set.seed(123)

svm = train(sentiment ~., data = x_train,
            method = "svmLinear",
            trControl = train_control,
            preProcess = c("center","scale"),
            metric = "ROC"
            )

print(str_c('Finished in: ',Sys.time() - start_time))
```

    ## [1] "Finished in: 3.5228223323822"

``` r
stopCluster(cl)
```

predict classes based on test set

``` r
# compute predictions
predicted_classes = svm %>% predict(x_test %>% select(-sentiment))
predicted_classes_prob = svm %>% predict(x_test %>% select(-sentiment), type='prob')
```

create metric

``` r
# transform to binary
binary_test = x_test %>% mutate(sentiment= ifelse(sentiment == 'p', 1, 0))
binary_pred = tibble(sentiment = predicted_classes) %>% mutate(sentiment=ifelse(sentiment == 'p', 1, 0))

# compute pROC
pROC_obj = pROC::roc(binary_pred$sentiment, binary_test$sentiment)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

``` r
# init prob df for log loss calc
prob_df <- tibble(
  obs =  factor(x_test$sentiment),
  pred = predicted_classes,
  p = predicted_classes_prob$p,
  n = predicted_classes_prob$n)

metric_df = tibble(
  auc = pROC_obj$auc,
  loss = Metrics::mse(binary_test$sentiment, binary_pred$sentiment),
  roc = svm$results$ROC,
  mnLogLoss = mnLogLoss(prob_df, lev = c("p", "n"))
)
```

## Step 4: Report the results in a table or a nice plot.

table with results of AUC and LOSS

``` r
metric_df
```

    ## # A tibble: 1 × 4
    ##   auc        loss   roc mnLogLoss
    ##   <auc>     <dbl> <dbl>     <dbl>
    ## 1 0.7991448 0.201 0.881      1.37

AUC plot

``` r
plot(
    pROC_obj,
    auc.polygon=T,
    auc.polygon.col='lightblue',
    print.auc=T,
    identity.col='red',
    max.auc.polygon=T,
    grid=T
    )
```

![](assignment_7_files/figure-gfm/AUC%20plot-1.png)<!-- -->

confusion matrix
![](assignment_7_files/figure-gfm/confusion%20matrix-1.png)<!-- -->

tf idf matrix

``` r
as_tibble(train_tf_features[1:10, 1:10])
```

    ## # A tibble: 10 × 10
    ##      the    and      a     of     to     is     it   `in`      i   this
    ##    <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ##  1 0.242 0.217  0.248  0.0630 0.0634 0.132  0.0665 0.270  0.0368 0.0329
    ##  2 0.413 0.199  0.124  0.0506 0.153  0.0529 0.107  0.0541 0.148  0.106 
    ##  3 0.322 0.248  0.220  0.0839 0.113  0.117  0.0886 0.0897 0.0981 0.0292
    ##  4 0.283 0.124  0.248  0.126  0.169  0.176  0.0444 0.0450 0.0984 0.132 
    ##  5 0.364 0.275  0.137  0.140  0.141  0.104  0.0422 0.149  0.0467 0.0626
    ##  6 0.345 0.272  0.163  0.166  0.278  0.154  0.117  0.138  0.226  0.202 
    ##  7 0.147 0.301  0.0376 0.115  0.0385 0.120  0.242  0.0818 0.223  0.120 
    ##  8 0.184 0.126  0.157  0.160  0.161  0.134  0.101  0.137  0.187  0.200 
    ##  9 0.108 0.0369 0.184  0.0375 0.226  0.196  0.0396 0.160  0.0877 0.235 
    ## 10 0.348 0.214  0.261  0.0966 0.121  0.126  0.0510 0.181  0.0282 0.0757

## extra

Plotting words

``` r
library(wordcloud)

review_tf_idf <- movie_df %>% 
  unnest_tokens(word, review) %>% 
  count(sentiment, word, sort=T) %>% 
  bind_tf_idf(word, sentiment, n)

review_tf_idf %>% 
  filter(tf_idf > 0.000018) %>% 
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(tf_idf, word, fill=sentiment)) +
    geom_col(show.legend = F) +
    facet_wrap(~sentiment, ncol = 2, scales = "free") +
    labs(title='Words x Sentiment', subtitle = 'Displays the words with high tf idf for a sentiment') +
    theme_minimal()
```

![](assignment_7_files/figure-gfm/Words%20x%20Sentiment-1.png)<!-- -->

Positive sentiments

``` r
plot_cloud = function(word, freq, max_words=50){
  wordcloud(word, freq, 
          max.words = max_words, random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"))
}

positive_sentiments = review_tf_idf %>% 
  filter(tf_idf > 0.000002, sentiment==1) %>% 
  mutate(word = reorder(word, tf_idf))

plot_cloud(positive_sentiments$word, positive_sentiments$tf_idf, 100)
```

![](assignment_7_files/figure-gfm/Positive%20sentiments-1.png)<!-- -->

Negative sentiments

``` r
negative_sentiments = review_tf_idf %>% 
  filter(tf_idf > 0.000002, sentiment==0) %>% 
  mutate(word = reorder(word, tf_idf))

plot_cloud(negative_sentiments$word, negative_sentiments$tf_idf, 100)
```

![](assignment_7_files/figure-gfm/Negative%20sentiments-1.png)<!-- -->
