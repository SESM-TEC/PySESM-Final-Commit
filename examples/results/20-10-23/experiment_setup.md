This experiment yields four different results based on diagonal and non-diagonal covariance for multivariate normal probability density 
functions (PDFs). These functions serve as words in the dictionary (`l_functions`), which is used as a reference to create a surrogate
 model for them. For visualization, a 2D `x` vector consisting of 50 samples is utilized, with each dimension ranging from -2 to 2.

The results presented include training times and Mean Squared Error (MSE) values for each iteration of the SESM model, along with 
their standard deviation and mean values:

1. `resultados_1.csv-figs_1.zip`: Surrogate model for three PDFs with diagonal covariance.
2. `resultados_2.csv-figs_2.zip`: Surrogate model for three PDFs, two with diagonal covariance and one with non-diagonal covariance.
3. `resultados_3.csv-figs_3.zip`: Surrogate model for three PDFs, one with diagonal covariance and two with non-diagonal covariance.
4. `resultados_4.csv-figs_4.zip`: Surrogate model for three PDFs with non-diagonal covariance.

The following is a list of parameters required for this experiment (in the 2D particular case):

- `n_samples`: 50
- `n_features`: 2
- `l_functions`: 20

Hyperparameters for ista and dict layers:

- `ista_alpha`:v0.06
- `ista_lambd`: 0.005
- `dictionary_alpha`: 0.06

Epochs for the model, dictionary, and h layer:
- `m_epochs`: 25
- `dict_epochs`: 800
- `h_epochs`:1000

Total iterations for the experiment:
- `N_iter`: 11