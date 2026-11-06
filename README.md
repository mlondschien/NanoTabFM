# NanoTabFM

A small, readable PyTorch model for in-context binary classification on numerical
tables. Architecture is based on TabICLv2 and NanoTabICL. From `NanoTabFM`:

```
1. Standardize features.
2. Apply learned Fourier encoding to all entries of X_train and X_test.
   Apply first label embedding to y_test and add this to the encoded cells.
3. Apply induced self-attention (ISAB) to each column of the encoded feature matrix.
   This first "summarizes" the column's distribution with the inducing tokens, then
   relates each cell to the summarized column information.
4. Apply self-attention to each row of the encoded feature matrix.
5. Another ISAB.
6. Compress rows into a single embedding per row by having a learned summary token
   attend to the cells of its row.
7. Transformer.
8. Prediction head.

For defaults (embedding_size=96, num_heads=4, num_inducing_tokens=16,
num_context_layers=3, max_features=10, num_frequencies=16), approx. parameter counts
are 1. 4k, 2. 151k, 3. 75k, 4. 151k, 5. 75k, 6. 224k, 7. 19k, total 698k.
```

## Results

On TabArena with datasets subsampled 1'000 rows and 10 features, NanoTabFM achieves average ROC 0.75, comparable to a random forest (0.748) and better than logistic regression (0.73).