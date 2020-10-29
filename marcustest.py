import numpy as np
import matplotlib.pyplot as plt

import time

import make_idun_files
import run_paralell_local

idun = True

if idun:
    do_tests = make_idun_files.do_tests
else:
    do_tests = run_paralell_local.do_tests


def test1():
    i = [0]
    I = [10]
    d = [2, 4, 8]
    K = [2, 4, 8]
    h = [1]
    tau = [0.05]
    it_max = [100000]
    tol = [1e-5]

    do_tests(
        1, i, I, d, K, h, tau, it_max, tol, folder="t1")


def test2():
    i = [2]
    I = [10]
    d = [2, 4, 8]
    K = [2, 4, 8]
    h = [1]
    tau = [0.05]
    it_max = [100000]
    tol = [1e-5]

    do_tests(
        1, i, I, d, K, h, tau, it_max, tol, folder="t2")


def test3():
    i = [2]
    I = [5, 10, 15, 20, 50, 100]
    d = [2]
    K = [4]
    h = [1]
    tau = [0.05]
    it_max = [100000]
    tol = [1e-10]

    do_tests(
        1, i, I, d, K, h, tau, it_max, tol, folder="t3")


def test4():
    i = [2]
    I = [100]
    d = [2]
    K = [4]
    h = [1]
    tau = [0.001, 0.005, 0.01, 0.05, 0.1]
    it_max = [1000000]
    tol = [1e-10]

    do_tests(
        1, i, I, d, K, h, tau, it_max, tol, folder="t4")


def test5():
    i = [2]
    I = [10]
    d = [2]
    K = [4]
    h = [0.01, 0.05, 0.1, 1.0, 10.0]
    tau = [0.05]
    it_max = [1000000]
    tol = [1e-10]

    do_tests(
        1, i, I, d, K, h, tau, it_max, tol, folder="t5")


# 2D function tests:


def test6():
    i = [0]
    I = [10]
    d = [2, 4, 8]
    K = [2, 4, 8]
    h = [1]
    tau = [0.005]
    it_max = [100000]
    tol = [1e-10]

    do_tests(
        2, i, I, d, K, h, tau, it_max, tol, folder="t6")


def test7():
    i = [0]
    I = [10]
    d = [4]
    K = [4]
    h = [1]
    tau = [0.001, 0.005, 0.01, 0.05, 0.1]
    it_max = [100000]
    tol = [1e-10]

    do_tests(
        2, i, I, d, K, h, tau, it_max, tol, folder="t7")


def test8():
    i = [0]
    I = [10, 20, 50]
    d = [4]
    K = [4]
    h = [1]
    tau = [0.001]
    it_max = [100000]
    tol = [1e-10]

    do_tests(
        2, i, I, d, K, h, tau, it_max, tol, folder="t8")


# 3d tests


def test9():
    i = [2]
    I = [10]
    d = [4, 8, 16]
    K = [2, 4, 8]
    h = [1]
    tau = [0.001]
    it_max = [100000]
    tol = [1e-10]

    do_tests(
        3, i, I, d, K, h, tau, it_max, tol, folder="t9")


def test10():
    i = [2]
    I = [10]
    d = [4]
    K = [4]
    h = [1]
    tau = [0.001, 0.005, 0.01]
    it_max = [100000]
    tol = [1e-10]

    do_tests(
        3, i, I, d, K, h, tau, it_max, tol, folder="t10")


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # test9()
    test10()
