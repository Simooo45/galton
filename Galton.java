import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Galton {
    private Random rn                           = new Random();
    private final double[] distribution         = {0.4761856614544038, 0.24943288860794455, 0.13065652949161874, 0.0684397666805937, 0.0358497327421709, 0.018778604896230646, 0.00983650294926533, 0.005152501519978572, 0.0026989542981191647, 0.001413751024641368, 0.0007405430914733744, 0.0003879071072419258, 0.000203191314025255, 0.00010643452858820759, 5.575193472288876e-05, 2.9203664135828375e-05, 1.529729870716281e-05, 8.012944767744859e-06, 4.197295553944142e-06, 2.198603694121975e-06, 1.1516601920645072e-06, 6.032561491332009e-07, 3.15994235083048e-07, 1.6552231875165257e-07, 8.670296784915498e-08, 4.541625981648194e-08};
    private final int N                         = this.distribution.length - 1;
    private final int NPins                     = (this.N*(this.N+1))/2;
    private final int NMix                      = this.N + this.NPins;
    private final int generations               = 20000;
    private final int startingPopulationSize    = 100000;
    private final int populationSize            = 50;
    private final double mutationRateMAX        = 0.2;
    private final double mutationRateMIN        = 0.0001;
    private final boolean keepBorders           = true;
    private Map<double[], Double> population    = new HashMap<>();
    
    public Galton(){
        for (double d : this.distribution){
            if (d == 0){
                System.out.println("C'è uno zero!");
            }
        }

    }

    private double[] createIndividual(){
        double[] result = new double[this.N];
        for (int i = 0; i < this.N; i++){
            result[i] = this.rn.nextDouble();
        }
        return result;
    }

    private double[] createIndividualPins(){
        double[] result = new double[this.NPins];
        for (int i = 0; i < this.NPins; i++){
            result[i] = (double) this.rn.nextInt(2);
        }
        return result;
    }

    private double[] createIndividualMix(){
        double[] result = new double[this.NMix];
        System.arraycopy(this.createIndividual(), 0, result, 0, this.N);  
        System.arraycopy(this.createIndividualPins(), 0, result, this.N, this.NPins); 
        if (keepBorders){
            int counter = 0;
            for (int i = 1; i <= this.N; i++){
                result[this.N + counter] = 1;
                result[this.N + counter + i - 1] = 1;
                counter += i;
            }
        }
        return result;
    }

    private double[][] createStartingPopulation(){
        double[][] result = new double[this.startingPopulationSize][this.NMix];
        for(int i = 0; i < this.startingPopulationSize; i++){
            result[i] = this.createIndividualMix();
        }
        return result;
    }

    private double[] galtonScore(double[] individual){
        int i, j;
        int pin;
        double[] lastRow = new double[this.N + 1];
        double left =0 ;
        double right = 0;
        double[] lastLastRow = new double[this.N + 1];
        double[] ind = new double[this.N];
        double[] board = new double[this.NPins + this.N + 1];
        System.arraycopy(individual, 0, ind, 0, this.N);
        System.arraycopy(individual, this.N, board, 0, this.NPins);
        for (i = this.NPins; i < board.length; i++){
            board[i] = 1;
        }
        double[] matrix = new double[this.N + 1];
        matrix[0] = 1;
        for (i = 1; i <= this.N; i++){
            System.arraycopy(lastRow, 0, lastLastRow, 0, this.N + 1);
            System.arraycopy(matrix, 0, lastRow, 0, this.N + 1);
            for (j = 0; j < i; j++){
                pin = (int) board[(i*(i+1)/2) + j - 1];
                lastRow[j] = lastRow[j] * (1 - pin);
                matrix[j] = matrix[j] - lastRow[j];
            }

            for (j = i; j >= 0; j--){
                left = 0;
                right = matrix[j] * ind[i-1] ;
                if (j > 0)
                    left = matrix[j-1] * (1 - ind[i-1]);

                matrix[j] =  left + right;
                if (i > 1 && j > 0){
                    matrix[j] += lastLastRow[j-1];
                }
            }

            
        }
        return matrix;
    }

    public double chiSquare(double[] individual){
        double result = 0;
        double[] galtonScore = this.galtonScore(individual);
        for (int i = 0; i < galtonScore.length; i++){
            result += Math.pow(1 - ((double)galtonScore[i]/ this.distribution[i]), 2);
        }
        return result;
    }

    public double fitnessFunctionChi(double[] individual){
        double d = this.chiSquare(individual);
        return 1 / d;  
    } 

    private double[][] chooseParentsMix(){
        double totalWeight = this.population.values()
                                            .stream()
                                            .mapToDouble(f -> f)
                                            .sum();
        double r;
        double[][] result = new double[2][this.NMix];
        for (int i = 0; i < 2; i++){
            r = rn.nextDouble() * totalWeight;
            for(double[] f : this.population.keySet()){
                r -= this.population.get(f);
                if (r <= 0.0){
                    result[i] = f;
                    break;
                }
            }
        }
        return result;
    }

    public double[][] crossoverMix(double[][] parents){
        int change;
        double[][] children = new double[2][this.NMix];
        for (int i = 0; i < parents[0].length; i++){
            change = this.rn.nextInt(2);
            if (change == 0){
                children[0][i] = parents[0][i];
                children[1][i] = parents[1][i];
            } else {
                children[0][i] = parents[1][i];
                children[1][i] = parents[0][i];
            }
        }
        return children;
    }

    public double[] mutateMix(double[] individual){
        double mutationRate = rn.nextDouble()*(mutationRateMAX - mutationRateMIN) + mutationRateMIN;
        double change;
        for (int i = 0; i < individual.length; i++){
            change = rn.nextDouble();
            if (change < mutationRate){
                if (i < this.N){
                    individual[i] = rn.nextDouble();
                } else {
                    individual[i] = rn.nextInt(2);
                } 
            }
        }
        if (keepBorders){
            int counter = 0;
            for (int i = 1; i <= this.N; i++){
                individual[this.N + counter] = 1;
                individual[this.N + counter + i - 1] = 1;
                counter += i;
            }
        }
        return individual;

    }

    public double[] findGaltonMix(){
        double[][] startingPopulation = this.createStartingPopulation();
        List<double[]> children = new ArrayList<>();
        Map<double[], Double> toKeep = new HashMap<>();
        double[][] bros = new double[2][this.NMix];
        List<Map.Entry<double[], Double>> entryList; 
        double best;
        double temp;
        double[] bestInd = new double[this.NMix];
        int i;
        PrintWriter fout;

        Comparator<Map.Entry<double[], Double>> comp = new Comparator<Map.Entry<double[], Double>>() {
            @Override
            public int compare(Map.Entry<double[], Double> entry1, Map.Entry<double[], Double> entry2) {
                return entry2.getValue().compareTo(entry1.getValue());
            }
        };

        for (double[] f : startingPopulation){
            this.population.put(f, this.fitnessFunctionChi(f));
        }
        System.out.println("Popolazione iniziale creata!");
        for (int gen = 0; gen < this.generations; gen++){
            for (i = 0; i < this.populationSize/2; i++){
                bros = this.crossoverMix(this.chooseParentsMix());
                children.add(this.mutateMix(bros[0]));
                children.add(this.mutateMix(bros[1]));
            }
            for (double[] child : children){
                if (!this.population.keySet().contains(child)){
                    this.population.put(child, this.fitnessFunctionChi(child));
                }
            }
            children.clear();
             // Converti la mappa in una lista di entry
            entryList = new ArrayList<>(this.population.entrySet());

            // Ordina la lista in base ai valori in ordine decrescente
            Collections.sort(entryList, comp);

            // Estrai i primi 100 elementi dalla lista ordinata
            int count = 0;
            toKeep.clear();
            for (Map.Entry<double[], Double> entry : entryList) {
                toKeep.put(entry.getKey(), entry.getValue());
                count++;
                if (count >= this.populationSize) {
                    break;
                }
            }
            this.population.clear();
            this.population.putAll(toKeep);

            best = Double.MIN_VALUE;
            for (double[] f : this.population.keySet()){
                temp = this.population.get(f);
                if (temp > best){
                    best = temp;
                    bestInd = f;
                }
            }
            System.out.println("#############################");
            System.out.println("Generazione: " + gen );
            System.out.println("Miglior chi: " + 1/best);
            if ((gen + 1) % 100 == 0){
                try {
                    fout = new PrintWriter(new BufferedWriter(new FileWriter("Java_result.log", true)));
                    fout.println("#############################");
                    fout.println("Generazione: " + (gen+1) );
                    fout.println("Miglior chi: " + 1/best);
                    fout.print("Migliori parametri: ");
                    StringJoiner joiner = new StringJoiner(", ");
                    for (i = 0; i < bestInd.length; i++ ){
                        joiner.add("" + bestInd[i]);
                    }
                    fout.println(joiner.toString());
                    fout.close();
                } catch (IOException e) {
                }
            }
        }
        return bestInd;
    }

    public static void main(String[] args) {
        File f = new File("Java_result.log"); 
        f.delete();
        Galton galton = new Galton();
        
        double[] best = galton.findGaltonMix();

        double[] gScore = galton.galtonScore(best);
        StringJoiner joiner = new StringJoiner(", ");
        for (int i = 0; i < gScore.length; i++ ){
                joiner.add("" + gScore[i]);
        }
        System.out.println(joiner.toString());

    }
}