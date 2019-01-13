import matplotlib.pyplot as plt
import seaborn
import random
import math


def p_y_given_x(x, rho, m1, m2, s1, s2):
    mean = m2 + rho * s2 / s1 * (x-m1)
    cov = math.sqrt(1 - rho ** 2) * s2
    return random.normalvariate(mean, cov)

def p_x_given_y(y, rho, m1, m2, s1, s2):
    mean = m1 + rho * s1 / s2 * (y-m2)
    cov = math.sqrt(1 - rho**2) * s1
    return random.normalvariate(mean, cov)


def main():
    print("Gibbs sampling world!")
    N = 5000
    K = 200
    x_res = []
    y_res = []
    m1 = 10
    m2 = -5
    s1 = 5
    s2 = 2
    rho = 0.5
    y = m2
    for i in range(N):
        for j in range(K):
            x = p_y_given_x(y, rho, m1, m2, s1, s2)
            y = p_x_given_y(x, rho, m1, m2, s1, s2)
            x_res.append(x)
            y_res.append(y)
    
    ax = seaborn.distplot(x_res, bins=50, hist=True, color='green')
    seaborn.distplot(y_res, bins=50, hist=True, color='blue', ax=ax)
    plt.show()


if __name__ == "__main__":
    main()