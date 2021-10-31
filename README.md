Sentiment analysis
================
Koray Poyraz
10/29/2021

# Text Mining

## Step 1: Load data set and split it into training (80%) and test (20%) sets.

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

## Step 2: Create a TFiDF representation + run a SVM classifier

create tf idf vectors for train set

``` r
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

start_time <- Sys.time()

# the maximum number of words to keep
tfv = TfIdfVectorizer$new(
  max_features=500,
  remove_stopwords = T
  )

# create tf idf vectors for train set
train_tf_features = tfv$fit_transform(train_df$review)

print(str_c('Finished in: ',Sys.time() - start_time))
```

    ## [1] "Finished in: 4.51136621634165"

``` r
stopCluster(cl)
```

create tf idf vectors for test set

``` r
# create tf idf vectors for test set
test_tf_features = tfv$transform(test_df$review)
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

# prepare train control with 10-fold
train_control = trainControl(
  method="repeatedcv",
  number=10,
  repeats=3,
  classProbs = T,
  summaryFunction = twoClassSummary,
  verboseIter = T
  )
```

train SVM model

``` r
x_train = x_train %>% mutate(sentiment= ifelse(sentiment == 1, 'p', 'n'))
x_test = x_test %>% mutate(sentiment= ifelse(sentiment == 1, 'p', 'n'))

cl <- makePSOCKcluster(5)
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
```

    ## Aggregating results
    ## Fitting final model on full training set

``` r
print(str_c('Finished in: ',Sys.time() - start_time))
```

    ## [1] "Finished in: 8.04030179977417"

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
  roc = svm$results$ROC,
  auc = pROC_obj$auc,
  log_loss = mnLogLoss(prob_df, lev = c("p", "n"))
)
```

## Step 4: Report the results in a table or a nice plot.

table with results of AUC and LOSS

``` r
metric_df
```

    ## # A tibble: 1 × 3
    ##     roc auc       log_loss
    ##   <dbl> <auc>        <dbl>
    ## 1 0.876 0.7904721     1.41

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
    ##         s  movie   film      t    one   like   just   good    can   time
    ##     <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ##  1 0.0348 0.0794 0      0.0402 0      0.139  0.0494 0      0      0     
    ##  2 0.0574 0      0.138  0      0.0686 0.0766 0.0814 0      0      0     
    ##  3 0.0567 0.129  0.136  0      0.0677 0      0      0      0      0     
    ##  4 0.194  0      0      0.149  0.232  0.0862 0.0917 0.0949 0      0.202 
    ##  5 0      0.347  0      0      0      0      0      0.149  0.149  0     
    ##  6 0      0      0      0      0.121  0      0      0      0      0.159 
    ##  7 0.126  0.0717 0.226  0      0      0.168  0      0.0922 0.0925 0     
    ##  8 0.117  0      0.140  0      0      0.0778 0      0      0      0     
    ##  9 0.0904 0      0.108  0.104  0      0      0      0      0      0.141 
    ## 10 0.170  0.129  0.0340 0.163  0.0676 0.0377 0.160  0      0      0.0442

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
