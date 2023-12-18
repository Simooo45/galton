import random
import numpy as np
import time
import os
import datetime
import json
from alive_progress import alive_bar

class Galton:
    def __init__(self, configuretion_file="configuration.json"):
        """
            Inizializzazione della classe
        """
        with open(configuretion_file) as f:
            self.config = json.load(f)
        self.distribution = self.config["distribution"]
        self.n = len(self.distribution) - 1
        self.population = self.create_starting_population()
        self.scores = {}
        self.population_size = self.config["genetic_algorithms"]["population_size"]


    def create_individual(self):
        """
            Creazione di un individuo randomico.
        """
        return [random.random() for _ in range(len(self.distribution)-1)]


    def create_starting_population(self):
        """
            Creazione della popolazione iniziale.
        """
        return [self.create_individual() for _ in range(self.config["genetic_algorithms"]["starting_population_size"])]
    

    def galton_score(self, individual):
        """
            Calcola la probabilità della distribuzione di Galton usando iterazioni matriciali.
        """
        matrix = np.matrix(np.ones(1))
        zeros = np.matrix(np.zeros(1))
        for idx in range(self.n):
            probs = np.matrix([individual[idx], 1 - individual[idx]], dtype='float64')
            temp_matrix = np.matrix(np.dot(matrix, probs), dtype='float64')            
            col0 = np.concatenate((temp_matrix[:, 0], zeros))
            col1 = np.concatenate((zeros, temp_matrix[:, 1]))
            matrix = (col0+col1)
        return matrix.flatten().tolist()[0]


    def chi_square(self, individual):
        """
            Calcola chi^2 partendo dai valori di galton e la distribuzione attesa.
        """
        return sum([(e-s)**2/(e**2) for s, e in zip(self.galton_score(individual), self.distribution)])
    

    def test(self, individual):
        print(self.galton_score(individual))


    def fitness_function(self, individual):
        """
            Inverte il chi quadro per ottenere valori crescenti (utili a random.choices)
        """
        return 1/self.chi_square(individual)
    

    def choose_parents(self, weights, bar=None):
        """
            Seleziona i genitori con le performances migliori.
        """
        if bar:
            bar()
        return random.choices(
                                self.population, 
                                weights=weights, 
                                k=2
                            )

    
    def crossover(self, parents, bar=None):
        """
            Ricombina i valori dei genitori per ottenere due figli in maniera randomica.
            Genera un terzo figlio come media dei genitori.
        """
        change = random.randint(0, self.n)
        combine = [True for _ in range(change)] + [False for _ in range(self.n - change)]
        random.shuffle(combine)

        child1 = [parents[0][idx] if combine[idx] else parents[1][idx] for idx in range(self.n)]
        child2 = [parents[1][idx] if combine[idx] else parents[0][idx] for idx in range(self.n)]
        child3 = [(parents[0][idx] + parents[1][idx])/2 for idx in range(self.n)]
        
        if bar:
            bar()
        return child1, child2, child3
    
   
    def mutate(self, individual):
        """
            Parte dei geni dell'individuo vengono mutati.
        """
        mutated_individual = []
        for i in range(self.n):
            if random.random() <= self.config["genetic_algorithms"]["mutation_rate"]:
                mutated_individual.append(random.random())
            else:
                mutated_individual.append(individual[i])
        return mutated_individual


    def find_galton(self):
        """
            Itera le generazioni per ottenere le performances migliori.
        """
        with alive_bar(len(self.population), title=f"Starting generation:".ljust(30)) as bar:
            for individual in self.population:
                self.scores[tuple(individual)]= self.fitness_function(individual)
                bar()
        print(f"Best of generation Starting generation:\t {1/(sorted(self.scores.values(), reverse=True)[0])}")
        
        # Esecuzione dell'algoritmo genetico
        for generation in range(self.config["genetic_algorithms"]["generations"]):

            # Selezione dei genitori
            weights=[self.scores[tuple(individual)] for individual in self.population]
            with alive_bar(self.population_size // 2, title = "Choosing parents: ".ljust(30)) as bar:
                parents = [self.choose_parents(weights, bar) for _ in range(self.population_size // 2)]

            # Crossover
            with alive_bar(len(parents), title=f"Crossover generation {generation}".ljust(30)) as bar:
                children = [self.crossover(parents_pair, bar) for parents_pair in parents]
                children = [child for sublist in children for child in sublist]  # Unflattening the list

            # Mutazione
            mutated_children = [self.mutate(child) for child in children]

            # Calcolo dei punteggi di fitness solo per i nuovi individui generati
            new_individuals = [individual for individual in mutated_children if tuple(individual) not in self.scores.keys()]
            with alive_bar(len(new_individuals), title=f"Generazione {generation+1}:".ljust(30)) as bar:
                for individual in new_individuals:
                    self.scores[tuple(individual)] = self.fitness_function(individual)
                    bar()
            print(f"Best of generation {generation + 1}:\t {1/(sorted(self.scores.values(), reverse=True)[0])}")

            self.scores = dict(sorted(self.scores.items(), key=lambda x:x[1])[-self.population_size:])

            f = open("results.log", "a")
            f.write(f"-- GENERETION: {generation} -- \t")
            f.write(f"{datetime.datetime.now()}\n")
            for key in dict(sorted(self.scores.items(), key=lambda x:x[1])[-5:]):
                f.write(f"{key}: {1/self.scores[key]},\n")
            f.close()

            # Nuova generazione: combinazione di individui originali e figli mutati
            self.population = self.population + mutated_children
            self.population = list(filter(lambda x: tuple(x) in list(self.scores.keys()), self.population))

        # Troviamo la soluzione migliore tra tutti gli individui
        best_solution = max(self.population, key=lambda individual: self.scores[tuple(individual)])
        best_fitness = self.scores[tuple(best_solution)]
        
        # Stampa su file
        f = open("best_galton.txt", "w")
        print("Miglior combinazione di parametri:", best_solution)
        f.write(f"Miglior combinazione di parametri: {best_solution} \n")
        print(f"Valore minimo della funzione: {1/best_fitness}")
        f.write(f"Valore minimo della funzione: {1/best_fitness} \n")

        return best_solution
    

    def simulate(self, best=None):
        """
            Simula la distribuzione di Galton di un individuo e la rappresenta graficamente su un file.
        """
        n_simulation = self.config["simulation"]["n_of_simulations"]
        if n_simulation <= 0:
            return
        
        n_marble = self.config["simulation"]["n_of_marbles"]
        step = self.config["simulation"]["height_step"]
        with open(self.config["simulation"]["simulation_file"], "w") as f:
            if not best:
                values = self.config["simulation"]["values"]
            else:
                values = best
            f.write(f"Simulazione dei parametri:\n{values}\nNumero palline: {n_marble}\n(# -> {step} {'palline' if step != 1 else 'pallina'})\n\n")
            if len(values) % 2 == 0:
                cols = list(range(-len(values)//2, +len(values)//2 +1))
            else:
                cols = [0.5*step for step in range(-len(values), +len(values)+1, 2)]

            for idx in range(1, n_simulation+1):
                f.write(f"SIMULAZIONE: {idx}\n\n")
                cols_dict = {col:0 for col in cols}
                for _ in range(n_marble):
                    column = sum([random.choices([-0.5, +0.5], weights=(values[i], 1 - values[i]), k=1)[0] for i in range(len(values))])
                    cols_dict[column] += 1
                
                max_height = max(cols_dict.values())
                graph = "\n".join(
                        ["|".join(
                            ['##' if cols_dict[column] > j else '  '
                                  for column in cols]) 
                            for j in range (0, max_height, step)][::-1]
                    ) 
                f.write(graph)
                f.write(f"\n{'-'.join(['--' for _ in cols])}\n\n")
                f.flush()
            




if __name__ == "__main__":
    galton = Galton()
    best = galton.find_galton()
    galton.test(best)
    galton.simulate(best)

