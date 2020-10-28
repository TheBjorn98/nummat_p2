Functions to test:

1. 1/2 y**2				(scalar)
2. 1 - cos(y)			(scalar)
3. 1/2 * norm(y)**2		(2D)
4. - 1 / norm(y)		(2D)

Additional functions for 3D counterparts:

1. 1/2 * norm(y)**2		(y in R3)
2. - 1 / norm(y)		(y in R3)

Parameters to vary:

1. K: amount of hidden layers
2. d: dimension of each hidden layer
3. tau: learning factor
4. h: stepsize between hidden layers
5. I: amount of parallell training data

Questions to parameter-variation:

1. Is tau and I correlated?
2. How large can tau be without breaking convergence?
3. To what extent does h affect the system at all.
4. How does K affect training time?
5. How does d affect training time?
6. Is (K, d) and I correlated? (since increasing K and d increases theta)