---
title: Report, project 2, numerical mathematics
date: 15.10.2020
---

## Some preliminary simplifications

From the project description we get that the elements in the gradient are given
as:

\begin{align*}
    \frac{\partial J}{\partial \mu}&=\eta^{\prime}\left(\left(Z^{(K)}\right)^{T} w+\mu \mathbf{1}\right)^{T}(\Upsilon-c)\\
    \frac{\partial \mathcal{J}}{\partial w}&=Z^{(K)}\left[(\Upsilon-c) \odot \eta^{\prime}\left(\left(Z^{(K)}\right)^{T} w+\mu\right)\right]\\
    \frac{\partial J}{\partial W_{k}} &=h\left[P^{(k+1)} \odot \sigma^{\prime}\left(W_{k} Z^{(k)}+b_{k}\right)\right] \cdot\left(Z^{(k)}\right)^{T} \\
    \frac{\partial J}{\partial b_{k}} &=h\left[P^{(k+1)} \odot \sigma^{\prime}\left(W_{k} Z^{(k)}+b_{k}\right)\right] \cdot \mathbf{1}
\end{align*}

Some simplifications can be made by renaming common terms and factors.
This makes it possible to calculate these terms and factors, and use them
wherever they're needed.
The simplifications we wish to introduce are:

\begin{gather*}
    Y_c = \Upsilon - c\\
    \nu = \eta'\left( \left(Z^{(K)}\right)^{T} w+\mu \right)\\
    H_k = h\left[P^{(k+1)} \odot \sigma^{\prime}\left(W_{k} Z^{(k)}+b_{k}\right)\right]
\end{gather*}

This reduces the partial derivatives to:

\begin{gather*}
    \frac{\partial J}{\partial \mu}&=\nu^{T}Y_c\\
    \frac{\partial \mathcal{J}}{\partial w}&=Z^{(K)}\left[Y_c \odot \nu\right]\\
    \frac{\partial J}{\partial W_{k}} &=H_k \cdot\left(Z^{(k)}\right)^{T} \\
    \frac{\partial J}{\partial b_{k}} &=H_k \cdot \mathbf{1}
\end{gather*}

These items can be computed with their definitions above, resulting in, hopefully,
more readable code.

## Gradient of the Neural Network

The aritficial neural network is a function approximator \(\tilde{F}(y; \theta): \mathbb{R}^{d} \rightarrow \mathbb{R}\) which can be defined as a composition of maps \(\Phi_k\) and \(G\).

The maps \(\Phi_k: \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}, 0 \leq k \leq K-1\) and \(G: \mathbb{R}^{d} \rightarrow \mathbb{R}\) are given as:

\begin{equation}
    \Phi_k(y) = y + h \sigma \left(W_k y + b_k\right),\qquad G(y) = \eta \left(w^T y + \mu\right)
    \label{eq:phi_and_g}
\end{equation}

\(\Phi_k\) is the transformation from one hidden layer to the next, while \(G\) is the transformation collapsing the dimensions on the hidden layers into the correct terminal dimension.
The function approximator as a composition is therefore:

\begin{equation}
    \tilde{F}(y; \theta) = G \circ \Phi_{K - 1} \circ \Phi_{K - 2} \circ \dots \circ \Phi_1 \circ \Phi_0 (y)
    \label{eq:func_ann}
\end{equation}

From the forward sweep of the ANN, we get a set of \(Z^{(k)}\) which are the result of the succesive transformation of the input \(Y\) to the k-th hidden layer.
Alternatively, the forward sweep can be written as a composition of maps:

\begin{equation}
    Z^{(k)}(Y) = \Phi_{k - 1} \circ \dots \Phi_0 (Y)
    \label{eq:z_as_comp}
\end{equation}

From \cref{eq:z_as_comp} it is apparent that \(\tilde{F}(Y) = G \circ Z^{(K)}\) and \(Z^{(K)} = \Phi_{K-1} \circ Z^{(K-1)} (Y)\).

\begin{equation}
    \nabla_y \tilde{F}(Y) = \left(DZ^{(K)}\right)^T \nabla G \left(Z^{(K)}\right)
    \label{eq:nabla_y_F}
\end{equation}

\begin{equation}
    DZ^{(k)} = D\Phi_k\left(Z^{(k)}\right) DZ^{(k-1)} \Rightarrow \left(DZ^{(k)}\right)^T = \left(DZ^{(k-1)}\right)^T \left(D\Phi_k\left(Z^{(k)}\right)\right)^T
    \label{eq:dz}
\end{equation}

Note that the input \(Y\) is just a series of completely intependent vectors \(y_i\), and the gradient is thus computed "per column".
The gradient of the ANN with respect to the input \(y\) is therefore given in \cref{eq:nabla_y_F}.
Together with \cref{eq:dz}, this can be summarized into a matrix multiplication of Jacobian matrices.

\begin{equation}
    \nabla_y \tilde{F}(Y; \theta) = \left(D\Phi_0\left(Z^{(0)}\right)\right)^T \dots \left(D\Phi_{K-1}\left(Z^{(K-1)}\right)\right)^T \nabla G \left(Z^{(K)}\right)
\end{equation}

It is now necessary to compute \(\nabla G(Z^{(K)})\) and \((D\Phi_k)^T\) using the expressions from \cref{eq:phi_and_g}:

\begin{gather*}
    G(y) = \eta \left(\sum_{i=1}^d w_i y_i + \mu\right)
    \\
    \Rightarrow \frac{\partial G}{\partial y_j} = \eta' \left(\sum_{i=1}^d w_i y_i + \mu\right) w_j
    \\
    \therefore \nabla G(y) = \eta \left(w^T y + \mu\right) w
\end{gather*}

\begin{gather*}
    \Phi_k(y) = y + h \sigma\left(W_k y + b_k\right)
    \\
    \left[\Phi_k(y)\right]_i = y_i + h \sigma \left(\sum_{j = 1}^d (W_k)_{ij} y_j + b_i\right)
    \\
    \Rightarrow \frac{\partial (\Phi_k)_i}{\partial y_r} = \left[D\Phi_k\right]_{ir} = \delta_{ir} + h \sigma' \left(\sum_{j = 1}^d (W_k)_{ij} y_j + b_i\right) (W_k)_{ir}
    \\
    \therefore \left[(D\Phi_k)^T\right]_{ri} = \delta_{ir} + h \sigma' \left(\sum_{j = 1}^d (W_k)_{ij} y_j + b_i\right) (W_k)_{ir}
\end{gather*}

It might be noted that \((D\Phi_k) = \left[D\Phi_k\right]_{ri}\), meaning \((D\Phi_k)^T = \left[D\Phi_k\right]_{ir}\).
For sake of simplicity, let \(\sigma'_{k,i} = \sigma' \left(\sum_{j = 1}^d (W_k)_{ij} y_j + b_i\right)\)
Conveniently, multiplication of \((D\Phi_k)^{T}\) and a vector \(v\) gives an r-th component of:

\begin{gather*}
    \left[(D\Phi_k)^T v\right]_r = \sum_{i = 1}^d \left[D\Phi_k\right]_{ir} v_i = v_i \delta_{ir} + h \sum_{i = 1}^d \sigma'_{k,i} [W_k]_{ir} v_i
    \\
    \sigma'_{k,i} \cdot v_i \Rightarrow \sigma'_k \odot v,\qquad (W_k)_{ir} = W_k^T,\qquad v_i \delta_{ir} = v_r
    \\
    \left[(D\Phi_k)^T v\right]_r =  v_r + h \sum_{i = 1}^d (W_k)_{ir} (\sigma'_{k,i} \odot v_i)
    \\
    \therefore (D\Phi_k)^T v = v + W_k^T \left(h \sigma'(W_k y + b_k) \odot v\right)
\end{gather*}

Since \(Z^{(k)}\) are already known, it is possible to perform the computation of the gradient of the ANN.
The pseudocode for this implementation is given in % section for grad-comp
