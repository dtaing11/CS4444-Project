import numpy as np 
from typing import List
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


class Coordinate: 
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class monteCarlos: 
    def __init__(self, n: int ):
        self.n = n
    def generate_random_dot(self) -> float:
        return np.random.uniform(-1,1)
    def random_Coordinate(self) -> List[Coordinate]: 
        randomC = []
        for _ in range(self.n): 
            randomC.append(Coordinate(self.generate_random_dot(),self.generate_random_dot()))
        return randomC
    def check_dot_in_circle(self) -> int:
        all_dot = self.random_Coordinate()
        nCircle = 0 
        for i in all_dot: 
            nCircle += np.count_nonzero(i.x * i.x + i.y * i.y <= 1.0 )
        return nCircle
    def estimate_pi(self) -> float:
         return 4 * (self.check_dot_in_circle()/self.n)
    





##Convergence 
listEstimatePi = []
listOfN = []
truePi = math.pi

for i in range(6): 
    mC = monteCarlos(10 ** i)
    listEstimatePi.append(mC.estimate_pi())
    listOfN.append(10** i)


plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.axhline(truePi, linestyle="--", label="True pi")
plt.plot(listOfN, listEstimatePi, marker="o", label="Mean of π̂ over R runs")
plt.title("Convergence of Monte Carlo π")
plt.xlabel("Samples per run (log scale)")
plt.ylabel("Estimate of π")
plt.legend()
plt.tight_layout()
plt.show()



R = 500
n_list = [10**3, 10**4, 10**5]

def run_many_estimates(n, R):
    return [monteCarlos(n).estimate_pi() for _ in range(R)]

for n in n_list:
    estimates = run_many_estimates(n, R)
    mean_est = np.mean(estimates)
    std_est = np.std(estimates, ddof=1)

    x = np.linspace(min(estimates), max(estimates), 500)
    pdf = norm.pdf(x, loc=mean_est, scale=std_est)

    plt.figure(figsize=(9, 6))
    plt.hist(estimates, bins=30, density=True, alpha=0.6, edgecolor='black', label='Histogram of π estimates')
    plt.plot(x, pdf, 'r--', label=f'Normal PDF (μ={mean_est:.5f}, σ={std_est:.5f})')
    plt.axvline(truePi, color='green', linestyle='-', label='True π')
    plt.axvline(mean_est, color='black', linestyle='--', label='Mean estimate')
    plt.title(f"Sampling Distribution of Monte Carlo π̂ (n={n}, R={R})")
    plt.xlabel("π̂ estimate")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    