import numpy as np
import random


# Function scaling all input between the initial and terminal values alpha and
# beta. Infix operations are element-wise per numpy.
def scaleInput(ys, alpha, beta):
    a = np.min(ys)
    b = np.max(ys)

    return 1 / (b - a) * (alpha * (b - ys) + beta * (ys - a))


# Partitions the interval [a, b] into points where the distance between two
# neighboring points is a random number between ld and hd
# Spacing between ld and hd is drawn from a uniform distribution
def getRandomInput(a, b, ld, hd):
    ys = [a]

    while b - ys[-1] > hd:
        ys.append(ys[-1] + random.uniform(ld, hd))

    ys.append(b)

    return np.array(ys)


def concatflat(tup):
    return np.concatenate([np.array(x).flatten() for x in tup])


def reconstruct_flat(shapes, flat):
    reconstructed = []
    i = 0
    for shape in shapes:
        if shape == ():
            reconstructed.append(flat[i])
            i += 1
        else:
            size = np.prod(shape)
            reconstructed.append(np.reshape(flat[i:i + size], shape))
            i += size
    return reconstructed


if __name__ == "__main__":

    def _test_scaleInput():
        y1 = np.array([1, 2, 3, 4, 5])
        y2 = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

        print(y1)
        print(y2)

        t1 = scaleInput(y1, .2, .8)
        print(t1)

        t2 = scaleInput(y2, .2, .8)
        print(t2)

    def _test_randomInput():
        y1 = getRandomInput(0, 1, .1, .2)
        y2 = getRandomInput(0, 10, .1, .5)

        print(y1)
        print(y2)

    _test_randomInput()
