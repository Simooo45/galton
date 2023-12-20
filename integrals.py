from distributions_db import *
import scipy.integrate as spi
import numpy as np
import random


class distribution:
    def __init__ (self, distribution, left=None, right=None, phi_z=5e-12):
        self.distribution = distribution
        self.phi_z = phi_z
        self.range=10
        self.right = right if right else self.find_right()
        self.left = left if left else self.find_left()
    
    def find_left(self):
        min = -self.range
        max = +self.range
        
        for _ in range(10000):
            value = random.uniform(min, max)
            integral, error = spi.quad(self.distribution, -np.inf, value)
            if (integral > self.phi_z):
                max = value
            else:
                min = value
        return (max+min)/2

    def find_right(self):
        min = -self.range
        max = +self.range
        
        for _ in range(10000):
            value = random.uniform(min, max)
            integral, error = spi.quad(self.distribution, value, np.inf)
            if (integral < self.phi_z):
                max = value
            else:
                min = value

        return (max+min)/2
    
    def get_integrated_distribution(self, slices=26):
        step=abs(self.right-self.left)/slices
        return [spi.quad(self.distribution, self.left+step*i, self.left+step*(i+1))[0] 
                        for i in range (slices)]  

if __name__ == "__main__":
    dist = distribution(ESPONENTIAL, left=0)
    print(dist.get_integrated_distribution())
