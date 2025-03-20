# Query_Optimization_using_RL

The dataset used here is TREC Robust dataset. Link: https://ir-datasets.com/trec-robust04.html

Here the objective is given documents labeled as relevant or non relevant (taken as ground truth, can be found in qrels file inside the dataset), find the best weighted query so that the retrived documents we get using this query best matches with the ground truth.

We have used PyLucene for making index of the documents and retriving the documents using a query.

## Finding the suitable wiighted query

### terms of the query
The terms in our constructed query is just the important words from the relevant documents, can be found by Roccio Algorithm or finding the most frequent terms in the relevant documents after filtering the stopwords.

### Weights of the terms in the query
Once we have found the terms of the query, those are fixed. Next our goal is to find the suitable weights for the terms present in the query. For this purpose, we have used Reinforcement Learning.
