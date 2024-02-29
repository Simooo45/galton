import random
import numpy as np
import datetime
import json
import math
import time
from alive_progress import alive_bar
from bayes_opt import BayesianOptimization

class Galton:
    def __init__(self, configuretion_file="configuration.json"):
        """
            Inizializzazione della classe
        """
        with open(configuretion_file) as f:
            self.config = json.load(f)
        self.distribution = self.config["distribution"]
        self.np_distribution = np.array(self.distribution)
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
        matrix = np.array(np.ones(1))
        zeros = np.array(np.zeros(1))
        for idx in range(len(individual)):
            probs = np.array([[individual[idx], 1 - individual[idx]]], dtype='float64')
            temp_matrix = np.dot(matrix, probs)
            if idx == 0:
                col0 = np.concatenate(([temp_matrix[0]], zeros))
                col1 = np.concatenate((zeros, [temp_matrix[1]]))
            else:     
                col0 = np.concatenate((temp_matrix[:, 0], zeros))
                col1 = np.concatenate((zeros, temp_matrix[:, 1]))
            
            matrix = np.array([col0+col1]).T
        return matrix.flatten()
    

    def test(self, individual):
        print(self.galton_score(individual))

############################################ Fitness function #############################################################

    def chi_square(self, individual):
        """
            Calcola chi^2 partendo dai valori di galton e la distribuzione attesa.
        """
        return np.sum(np.square(np.ones(self.n+1)-(self.galton_score(individual)/self.np_distribution)))
    
    def fitness_function_chi(self, individual):
        """
            Inverte il chi quadro per ottenere valori crescenti (utili a random.choices)
        """
        return 1/self.chi_square(individual)
    
    def logarithm(self, individual):
        """
            Calcola la performance con una funzione logaritmica.
        """
        galton = self.galton_score(individual)
        log = np.abs(np.log(galton/self.np_distribution))
        result = np.sum(log)
        return result

    def fitness_function_log(self, individual):
        """
            Inverte la funzione logaritmica per ottenere valori crescenti (utili a random.choices)
        """
        return 1/self.logarithm(individual)

    
############################################ Genetic Algorithms ###########################################################
    
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


    def find_galton(self, function="chi"):
        """
            Itera le generazioni per ottenere le performances migliori.
        """
        if function == "log":
            fitness_function = self.fitness_function_log
        else:
            fitness_function = self.fitness_function_chi
        
        with alive_bar(len(self.population), title=f"Starting generation:".ljust(30)) as bar:
            for individual in self.population:
                self.scores[tuple(individual)]= fitness_function(individual)
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
                    self.scores[tuple(individual)] = fitness_function(individual)
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
    
############################################# Bayesian Optimization ##################################################################
    
    def proxy_fitness_function_log(self, **kwargs):
        return self.fitness_function_log(list(kwargs.values()))
    
    def proxy_fitness_function_chi(self, **kwargs):
        return self.fitness_function_chi(list(kwargs.values()))
    
    def bayesian_optimization(self, function="chi"):
        if function == "log":
            fitness_function = self.proxy_fitness_function_log
        else:
            fitness_function = self.proxy_fitness_function_chi

        # Inizializza l'ottimizzatore
        optimizer = BayesianOptimization(
            f=fitness_function,
            pbounds={f'{i}'.zfill(2): (0, 1) for i in range(1, self.n+1)}
        )

        # Esegui le iterazioni di ottimizzazione
        optimizer.maximize(
            init_points=self.config["bayesian"]["init_points"],  # Numero di punti iniziali casuali
            n_iter=self.config["bayesian"]["generations"]  # Numero di iterazioni dell'ottimizzazione bayesiana
        )

        # I risultati dell'ottimizzazione sono accessibili tramite optimizer.max
        migliori_parametri = optimizer.max['params']
        valore_massimizzato = optimizer.max['target']
        print(migliori_parametri)
        print(list(migliori_parametri.values()))
        print(1/valore_massimizzato)
        return list(migliori_parametri.values())

################################################ Simulation ###########################################################################

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
                with alive_bar(n_marble, title=f"Simulazione {idx}:") as bar:
                    for _ in range(n_marble):
                        column = sum([random.choices([-0.5, +0.5], weights=(values[i], 1 - values[i]), k=1)[0] for i in range(len(values))])
                        cols_dict[column] += 1
                        bar()

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
        print("Simulazione terminata")




if __name__ == "__main__":
    galton = Galton()
    start = time.time()
    print(sum(galton.distribution))
    # print(galton.chi_square([0.5450393372305553, 0.5350242393785283, 0.5013578056935863, 0.48477388840143615, 0.5203916447561663, 0.49118258724747116, 0.4875636662459224, 0.5171323066885223, 0.5062444829004319, 0.550533897613901, 0.4998497915084896, 0.4976310196705649, 0.46686509502843954, 0.48513409114897943, 0.45743970311938975, 0.428910601701293, 0.4725467030455688, 0.5369286156474097, 0.48039425062781355, 0.5309034078785204, 0.5410756982687943, 0.47188019903285805, 0.4983719878762516, 0.45643649854332596, 0.5442642189090006]))
    # best = galton.bayesian_optimization("chi")
    best = galton.find_galton("chi")
    galton.test(best)
    # galton.simulate(best)
    end = time.time()
    print(f"{end-start}s")
