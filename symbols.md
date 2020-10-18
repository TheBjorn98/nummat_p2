# Symbols used in the project description

## Vectors and matrices used in calculations

* $z^{(k)}$: d x 1 vector
* $Z^{(k)}$: d x I matrix
* $b_k$: d x 1 vector
* $b$: d x K matrix (collection of all $b_k$)
* $W_k$: d x d matrix
* $W$: d x d x K collection of all $W_k$
* $w$: d x 1 vector of weights for dotting with $z^{(K)}$
* $y_i$: d x 1 vector of input data (to the function and function approximator)
* $Y$: d x I matrix holding I sets of input data
* $c_i$: scalar, result of $F(y_i)$
* $c$: I x 1 vector of function values (exact or approximate)

## Functions on vectors and matrices

* $\eta(x)$: hypothesis function
* $\eta'(x)$: derivative of hypothesis wrt. $x$
* $\sigma(x)$: activation function
* $\sigma'(x)$: derivative of activation wrt. $x$