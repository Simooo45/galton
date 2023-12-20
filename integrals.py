from distributions_db import *
import scipy.integrate as spi
import numpy as np
import random


class distribution:
    def __init__ (self, distribution, left=None, right=None, phi_z=5e-12):
        self.distribution = distribution
        self.phi_z = phi_z
        self.range=1
        self.right = right if right != None else self.find_right()
        self.left = left if left != None else self.find_left()
    
    def find_left(self):
        min = -self.range
        max = +self.range

        integral, error = spi.quad(self.distribution, -np.inf, min)
        
        # Mi assicuro che min sia inizialmente a destra di phi_z
        while integral < self.phi_z:
            min, max = min+1, max+1
            integral, error = spi.quad(self.distribution, -np.inf, min)

        # Pongo min a sinistra di phi_z
        while integral > self.phi_z:
            min, max = min-1, max-1
            integral, error = spi.quad(self.distribution, -np.inf, min)

        # Pongo max a destra di phi_z
        integral, error = spi.quad(self.distribution, -np.inf, max)
        while integral > self.phi_z:
            max += 1
            integral, error = spi.quad(self.distribution, -np.inf, max)

        for _ in range(10000):
            value = random.uniform(min, max)
            integral, error = spi.quad(self.distribution, -np.inf, value)
            if (integral > self.phi_z):
                max = value
            else:
                min = value
        result = (max+min)/2
        print(f"Estremo sinistro di integrazione: {result}")
        return result

    def find_right(self):
        min = -self.range
        max = +self.range

        # Mi assicuro che max sia inizialmente a sinistra di phi_z
        integral, error = spi.quad(self.distribution, max, np.inf)
        while integral < self.phi_z:
            min, max = min-1, max-1
            integral, error = spi.quad(self.distribution, max, np.inf)

        # Pongo max a destra di phi_z
        while integral > self.phi_z:
            min, max = min+1, max+1
            integral, error = spi.quad(self.distribution, max, np.inf)

        # Pongo min a sinistra di phi_z
        integral, error = spi.quad(self.distribution, min, np.inf)
        while integral > self.phi_z:
            min -= 1
            integral, error = spi.quad(self.distribution, min, np.inf)
        
        for _ in range(10000):
            value = random.uniform(min, max)
            integral, error = spi.quad(self.distribution, value, np.inf)
            if (integral < self.phi_z):
                max = value
            else:
                min = value

        result = (max+min)/2
        print(f"Estremo destro di integrazione: {result}")
        return result
    
    def get_integrated_distribution(self, slices=26):
        step=abs(self.right-self.left)/slices
        return [spi.quad(self.distribution, self.left+step*i, self.left+step*(i+1))[0] 
                        for i in range (slices)]  

if __name__ == "__main__":
    dist = distribution(EXPONENTIAL, left=0, phi_z=5e-18)
    print(dist.get_integrated_distribution())
