import java.io.*;
import java.util.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;

public class GaltonBigDecimal {
    /*
     * DISTRIBUZIONE ESPONENZIALE 
     * {0.4761856614544038, 0.24943288860794455, 0.13065652949161874, 0.0684397666805937, 0.0358497327421709, 0.018778604896230646, 0.00983650294926533, 0.005152501519978572, 0.0026989542981191647, 0.001413751024641368, 0.0007405430914733744, 0.0003879071072419258, 0.000203191314025255, 0.00010643452858820759, 5.575193472288876e-05, 2.9203664135828375e-05, 1.529729870716281e-05, 8.012944767744859e-06, 4.197295553944142e-06, 2.198603694121975e-06, 1.1516601920645072e-06, 6.032561491332009e-07, 3.15994235083048e-07, 1.6552231875165257e-07, 8.670296784915498e-08, 4.541625981648194e-08};
     * DISTRIBUZIONE UNIFORME
     * {0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346}
     */
    private final double[] distributionDouble       = {0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346, 0.0384615346};
    private final int N                             = this.distributionDouble.length - 1;
    private final int NPins                         = (this.N*(this.N+1))/2;
    private final int NMix                          = this.N + this.NPins;
    private final int generations                   = 10000;
    private final int startingPopulationSize        = 100;
    private final int populationSize                = 50;
    private final BigDecimal mutationRateMAX        = BigDecimal.valueOf(0.2);
    private final BigDecimal mutationRateMIN        = BigDecimal.valueOf(0.0001);
    private final String fun                        = "chi";        // "log" o "chi"
    private final boolean keepBorders               = true;
    private final boolean initRandomBoard           = false;

    private BigDecimal[] distribution               = new BigDecimal[this.distributionDouble.length]; 
    private Random rn                                   = new Random();
    private Map<BigDecimal[], BigDecimal> population    = new HashMap<>();
    
    public GaltonBigDecimal(){
        for (int i = 0; i < this.distributionDouble.length; i++){
            if (this.distributionDouble[i] == 0){
                System.out.println("C'è uno zero!");
            }
            this.distribution[i] = BigDecimal.valueOf(this.distributionDouble[i]);
        }

    }

    private BigDecimal[] createIndividual(){
        BigDecimal[] result = new BigDecimal[this.N];
        for (int i = 0; i < this.N; i++){
            if (initRandomBoard) {
                result[i] = BigDecimal.valueOf(this.rn.nextDouble());
            } else {
                result[i] = BigDecimal.ONE;
            }
        }
        return result;
    }

    private BigDecimal[] createIndividualPins(){
        BigDecimal[] result = new BigDecimal[this.NPins];
        for (int i = 0; i < this.NPins; i++){
            result[i] = BigDecimal.valueOf(this.rn.nextInt(2));
        }
        return result;
    }

    private BigDecimal[] createIndividualMix(){
        BigDecimal[] result = new BigDecimal[this.NMix];
        System.arraycopy(this.createIndividual(), 0, result, 0, this.N);  
        System.arraycopy(this.createIndividualPins(), 0, result, this.N, this.NPins); 
        if (keepBorders){
            int counter = 0;
            for (int i = 1; i <= this.N; i++){
                result[this.N + counter] = BigDecimal.ONE;
                result[this.N + counter + i - 1] = BigDecimal.ONE;
                counter += i;
            }
        }
        return result;
    }

    private BigDecimal[][] createStartingPopulation(){
        BigDecimal[][] result = new BigDecimal[this.startingPopulationSize][this.NMix];
        for(int i = 0; i < this.startingPopulationSize; i++){
            result[i] = this.createIndividualMix();
        }
        return result;
    }

    private BigDecimal[] galtonScore(BigDecimal[] individual){
        int i, j;
        int pin;
        BigDecimal[] lastRow = new BigDecimal[this.N + 1];
        BigDecimal left = BigDecimal.ZERO;
        BigDecimal right = BigDecimal.ZERO;
        BigDecimal[] lastLastRow = new BigDecimal[this.N + 1];
        BigDecimal[] ind = new BigDecimal[this.N];
        BigDecimal[] board = new BigDecimal[this.NPins + this.N + 1];
        System.arraycopy(individual, 0, ind, 0, this.N);
        System.arraycopy(individual, this.N, board, 0, this.NPins);
        for (i = this.NPins; i < board.length; i++){
            board[i] = BigDecimal.ONE;
        }
        BigDecimal[] matrix = new BigDecimal[this.N + 1];
        matrix[0] = BigDecimal.ONE;
        for (i = 1; i < matrix.length; i++){
            matrix[i] = BigDecimal.ZERO;
        }
        for (i = 1; i <= this.N; i++){
            System.arraycopy(lastRow, 0, lastLastRow, 0, this.N + 1);
            System.arraycopy(matrix, 0, lastRow, 0, this.N + 1);
            for (j = 0; j < i; j++){
                pin = (int) board[(i*(i+1)/2) + j - 1].doubleValue();
                lastRow[j] = lastRow[j].multiply(BigDecimal.valueOf(1 - pin));
                matrix[j] = matrix[j].subtract(lastRow[j]);
            }

            for (j = i; j >= 0; j--){
                left = BigDecimal.ZERO;
                right = matrix[j].multiply(ind[i-1]);
                if (j > 0)
                    left = matrix[j-1].multiply(BigDecimal.ONE.subtract(ind[i-1]));

                matrix[j] =  left.add(right);
                if (i > 1 && j > 0){
                    matrix[j] = matrix[j].add(lastLastRow[j-1]);
                }
            }

            
        }
        return matrix;
    }

    public static BigDecimal logDecimal(int base_int, BigDecimal x) {
        BigDecimal result = BigDecimal.ZERO;

        BigDecimal input = new BigDecimal(x.toString());
        int decimalPlaces = 100;
        int scale = input.precision() + decimalPlaces;

        int maxite = 10000;
        int ite = 0;
        BigDecimal maxError_BigDecimal = new BigDecimal(BigInteger.ONE,
                decimalPlaces + 1);
        // System.out.println("maxError_BigDecimal " + maxError_BigDecimal);
        // System.out.println("scale " + scale);

        RoundingMode a_RoundingMode = RoundingMode.UP;

        BigDecimal two_BigDecimal = new BigDecimal("2");
        BigDecimal base_BigDecimal = new BigDecimal(base_int);

        while (input.compareTo(base_BigDecimal) == 1) {
            result = result.add(BigDecimal.ONE);
            input = input.divide(base_BigDecimal, scale, a_RoundingMode);
        }

        BigDecimal fraction = new BigDecimal("0.5");
        input = input.multiply(input);
        BigDecimal resultplusfraction = result.add(fraction);
        while (((resultplusfraction).compareTo(result) == 1)
                && (input.compareTo(BigDecimal.ONE) == 1)) {
            if (input.compareTo(base_BigDecimal) == 1) {
                input = input
                        .divide(base_BigDecimal, scale, a_RoundingMode);
                result = result.add(fraction);
            }
            input = input.multiply(input);
            fraction = fraction.divide(two_BigDecimal, scale,
                    a_RoundingMode);
            resultplusfraction = result.add(fraction);
            if (fraction.abs().compareTo(maxError_BigDecimal) == -1) {
                break;
            }
            if (maxite == ite) {
                break;
            }
            ite++;
        }

        MathContext a_MathContext = new MathContext(
                ((decimalPlaces - 1) + (result.precision() - result.scale())),
                RoundingMode.HALF_UP);
        BigDecimal roundedResult = result.round(a_MathContext);
        BigDecimal strippedRoundedResult = roundedResult
                .stripTrailingZeros();
        //return result;
        //return result.round(a_MathContext);
        return strippedRoundedResult;
    }


    public BigDecimal log(BigDecimal[] individual){
        BigDecimal result = BigDecimal.ZERO;
        BigDecimal[] galtonScore = this.galtonScore(individual);
        for (int i = 0; i < galtonScore.length; i++){
            result = result.add((logDecimal(10, galtonScore[i].divide(this.distribution[i]))).abs());
        }
        return result;
    }

    public BigDecimal chiSquare(BigDecimal[] individual){
        BigDecimal result = BigDecimal.ZERO;
        BigDecimal temp;
        BigDecimal[] galtonScore = this.galtonScore(individual);
        for (int i = 0; i < galtonScore.length; i++){
            temp = BigDecimal.valueOf(1.0).subtract(galtonScore[i].divide(this.distribution[i], 2, RoundingMode.HALF_UP));
            result = result.add(temp.multiply(temp));
        }
        return result;
    }

    public BigDecimal fitnessFunction(BigDecimal[] individual){
        BigDecimal d;
        if (this.fun.equals("log")){
            d = this.log(individual);
        } else {
            d = this.chiSquare(individual);
        }
        return  BigDecimal.ONE.divide(d, 2, RoundingMode.HALF_UP);  
    } 

    private BigDecimal[][] chooseParentsMix(){
        BigDecimal totalWeight = this.population.values()
                                            .stream()
                                            .reduce(BigDecimal.ZERO, BigDecimal::add);
        BigDecimal r;
        BigDecimal[][] result = new BigDecimal[2][this.NMix];
        for (int i = 0; i < 2; i++){
            r = BigDecimal.valueOf(rn.nextDouble()).multiply(totalWeight);
            for(BigDecimal[] f : this.population.keySet()){
                r = r.subtract(this.population.get(f));
                if (r.compareTo(BigDecimal.ZERO) <= 0){
                    result[i] = f;
                    break;
                }
            }
        }
        return result;
    }

    public BigDecimal[][] crossoverMix(BigDecimal[][] parents){
        int change;
        BigDecimal[][] children = new BigDecimal[2][this.NMix];
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

    public BigDecimal[] mutateMix(BigDecimal[] individual){
        BigDecimal mutationRate = BigDecimal.valueOf(rn.nextDouble()).multiply(mutationRateMAX.subtract(mutationRateMIN)).add(mutationRateMIN);
        BigDecimal change;
        for (int i = 0; i < individual.length; i++){
            change = BigDecimal.valueOf(rn.nextDouble());
            if (change.compareTo(mutationRate) < 0){
                if (i < this.N){
                    individual[i] = BigDecimal.valueOf(rn.nextDouble());
                } else {
                    individual[i] = BigDecimal.valueOf(rn.nextInt(2));
                } 
            }
        }
        if (keepBorders){
            int counter = 0;
            for (int i = 1; i <= this.N; i++){
                individual[this.N + counter] = BigDecimal.ONE;
                individual[this.N + counter + i - 1] = BigDecimal.ONE;
                counter += i;
            }
        }
        return individual;

    }

    public BigDecimal[] findGaltonMix(){
        BigDecimal[][] startingPopulation = this.createStartingPopulation();
        List<BigDecimal[]> children = new ArrayList<>();
        Map<BigDecimal[], BigDecimal> toKeep = new HashMap<>();
        BigDecimal[][] bros = new BigDecimal[2][this.NMix];
        List<Map.Entry<BigDecimal[], BigDecimal>> entryList; 
        BigDecimal best;
        BigDecimal temp;
        BigDecimal[] bestInd = new BigDecimal[this.NMix];
        int i;
        PrintWriter fout;

        Comparator<Map.Entry<BigDecimal[], BigDecimal>> comp = new Comparator<Map.Entry<BigDecimal[], BigDecimal>>() {
            @Override
            public int compare(Map.Entry<BigDecimal[], BigDecimal> entry1, Map.Entry<BigDecimal[], BigDecimal> entry2) {
                return entry2.getValue().compareTo(entry1.getValue());
            }
        };

        for (BigDecimal[] f : startingPopulation){
            this.population.put(f, this.fitnessFunction(f));
        }
        System.out.println("Popolazione iniziale creata!");
        for (int gen = 0; gen < this.generations; gen++){
            for (i = 0; i < this.populationSize/2; i++){
                bros = this.crossoverMix(this.chooseParentsMix());
                children.add(this.mutateMix(bros[0]));
                children.add(this.mutateMix(bros[1]));
            }
            for (BigDecimal[] child : children){
                if (!this.population.keySet().contains(child)){
                    this.population.put(child, this.fitnessFunction(child));
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
            for (Map.Entry<BigDecimal[], BigDecimal> entry : entryList) {
                toKeep.put(entry.getKey(), entry.getValue());
                count++;
                if (count >= this.populationSize) {
                    break;
                }
            }
            this.population.clear();
            this.population.putAll(toKeep);

            best = BigDecimal.valueOf(Double.MIN_VALUE);
            for (BigDecimal[] f : this.population.keySet()){
                temp = this.population.get(f);
                if (temp.compareTo(best) > 0){
                    best = temp;
                    bestInd = f;
                }
            }
            System.out.println("#############################");
            System.out.println("Generazione: " + gen );
            System.out.println("Miglior chi: " + BigDecimal.ONE.divide(best, 2, RoundingMode.HALF_UP));
            if ((gen + 1) % 100 == 0){
                try {
                    fout = new PrintWriter(new BufferedWriter(new FileWriter("Java_result.log", true)));
                    fout.println("#############################");
                    fout.println("Generazione: " + (gen+1) );
                    fout.println("Miglior chi: " + BigDecimal.ONE.divide(best, 2, RoundingMode.HALF_UP));
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
        GaltonBigDecimal galton = new GaltonBigDecimal();
        
        BigDecimal[] best = galton.findGaltonMix();

        BigDecimal[] gScore = galton.galtonScore(best);
        StringJoiner joiner = new StringJoiner(", ");
        for (int i = 0; i < gScore.length; i++ ){
                joiner.add("" + gScore[i]);
        }
        System.out.println(joiner.toString());
    }
}