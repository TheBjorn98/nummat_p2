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

