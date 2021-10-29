import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.execution.columnar.DOUBLE;
import scala.Tuple2;

import java.util.*;

public class G49HM4 {
    public static void main(String[] args) throws Exception {

        //------- PARSING CMD LINE ------------
        // Parameters are:
        // <path to file>, k, L and iter

        if (args.length != 4) {
            System.err.println("USAGE: <filepath> k L iter");
            System.exit(1);
        }
        String inputPath = args[0];
        int k = 0, L = 0, iter = 0;
        try {
            k = Integer.parseInt(args[1]);
            L = Integer.parseInt(args[2]);
            iter = Integer.parseInt(args[3]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (k <= 2 && L <= 1 && iter <= 0) {
            System.err.println("Something wrong here...!");
            System.exit(1);
        }
        //------------------------------------
        final int k_fin = k;

        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //------- SETTING THE SPARK CONTEXT
        SparkConf conf = new SparkConf(true).setAppName("kmedian new approach");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //------- PARSING INPUT FILE ------------
        JavaRDD<Vector> pointset = sc.textFile(args[0], L)
                .map(x -> strToVector(x))
                .repartition(L)
                .cache();
        long N = pointset.count();

        System.out.println("Number of points is: " + N);
        System.out.println("Number of clusters is: " + k);
        System.out.println("Number of parts is: " + L);
        System.out.println("Number of iterations is: " + iter);

        //------- SOLVING THE PROBLEM ------------
        double obj = MR_kmedian(pointset, k, L, iter);
        System.out.println("Objective function is: "+ obj  );

    }

    public static Double MR_kmedian(JavaRDD<Vector> pointset, int k, int L, int iter) {
        //
        // --- ADD INSTRUCTIONS TO TAKE AND PRINT TIMES OF ROUNDS 1, 2 and 3
        //

        //START Time
        long start = System.currentTimeMillis();

        //------------- ROUND 1 ---------------------------

        JavaRDD<Tuple2<Vector, Long>> coreset = pointset.mapPartitions(x ->
        {

            ArrayList<Vector> points = new ArrayList<>();
            ArrayList<Long> weights = new ArrayList<>();
            while (x.hasNext()) {
                points.add(x.next());
                weights.add(1L);
            }
            final ArrayList<Vector> centers = kmeansPP(points, weights, k, iter);

            ArrayList<Long> weight_centers = compute_weights(points, centers);

            ArrayList<Tuple2<Vector, Long>> c_w = new ArrayList<>();
            for (int i = 0; i < centers.size(); ++i) {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weight_centers.get(i));
                c_w.add(i, entry);
            }
            return c_w.iterator();
        });


        //------------- ROUND 2 ---------------------------

        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>(k * L);

        elems.addAll(coreset.collect());

        long step1 = System.currentTimeMillis();

        ArrayList<Vector> coresetPoints = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();


        for (int i = 0; i < elems.size(); ++i) {
            coresetPoints.add(i, elems.get(i)._1);
            weights.add(i, elems.get(i)._2);
        }

        ArrayList<Vector> centers = kmeansPP(coresetPoints, weights, k, iter);

        long step2 = System.currentTimeMillis();


        //------------- ROUND 3: COMPUTE OBJ FUNCTION --------------------
        //
        //------------- ADD YOUR CODE HERE--------------------------------
        //

        double obj = kmeansObj(pointset, centers);
        long step3 = System.currentTimeMillis();

        System.out.println("STAMPA DEI TEMPI ");
        System.out.println("TEMPO ROUND 1: " + (step1 - start));
        System.out.println("TEMPO ROUND 2: " + (step2 - step1));
        System.out.println("TEMPO ROUND 3: " + (step3 - step2));

        return obj;

    }

    public static double kmeansObj(JavaRDD<Vector> poinsets, ArrayList<Vector> centers) {

        // Compute the mean of the distances between each point and its closest center
        Double totalsum = poinsets.mapPartitions(v -> {
            ArrayList<Vector> points = new ArrayList<>();
            double currentDist;
            ArrayList<Double> dist = new ArrayList<>(); // array containing the distances of each point from the closest center

            while (v.hasNext()) {
                points.add(v.next()); // Add all the points of the partition to the array points
            }
            for (Vector p : points) {
                double minDist = Double.POSITIVE_INFINITY;
                for (Vector c : centers) {
                    currentDist = Math.sqrt(Vectors.sqdist(p, c));
                    if (currentDist < minDist) {
                        minDist = currentDist; // Compute the minimum distance
                    }
                }
                dist.add(minDist);
            }

            return dist.iterator();

        }).reduce((x, y) -> x + y); //Compute the sum of each distance
        return totalsum / poinsets.count();// Return the mean of the distances between each point and its closest center
    }

    public static ArrayList<Long> compute_weights(ArrayList<Vector> points, ArrayList<Vector> centers) {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for (int i = 0; i < points.size(); ++i) {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for (int j = 1; j < centers.size(); ++j) {
                if (euclidean(points.get(i), centers.get(j)) < tmp) {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // Euclidean distance
    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> WP, Integer k, Integer iter) {
        ArrayList<Vector> S = new ArrayList<>();    // create set of centers
        ArrayList<Vector> P_S = new ArrayList<>();
        ArrayList<Vector> closestCenters = new ArrayList<>(); // array containing closest centers of each points
        P_S.addAll(P); //create P-S
        ArrayList<Long> WPcenters = new ArrayList<>(); // array containing weigths of the centers


        //extract a random point with uniform probability
        int bound = (int) (Math.random() * P.size());
        S.add(P_S.remove(bound)); //select the first center randomly and add it to set of centers S and we remove it from P-S
        WPcenters.add(WP.remove(bound));

        ArrayList<Double> arrayDistances = new ArrayList<>();
        //iterate for number of clusters
        for (int i = 1; i < k; i++) {

            //Initialize the usefull variable to compute minimum distance
            double dist;
            double denom = 0;

            Vector lastcenter = S.get(S.size() - 1);

            for (int j = 0; j < P_S.size(); j++) {//Iterate for each point of P-S

                dist = Math.sqrt(Vectors.sqdist(P_S.get(j), lastcenter));
                if (i == 1) {
                    arrayDistances.add(dist);
                    closestCenters.add(lastcenter);
                } else {
                    if (dist < arrayDistances.get(j)) {
                        arrayDistances.set(j, dist);
                        closestCenters.set(j, lastcenter);
                    }
                }

                denom += (WP.get(j) * arrayDistances.get(j)); // Compute the denominator of probability
            }

            double randomX = Math.random();    // Compute random number x between 0 and 1
            double currentSumProbability = 0;  // Variable used to compute the sum of probability including the last point
            double previousSumProbability = 0; // Variable used to compute the sum of probability excluding the last point


            //Iterate P-S and select the point with higher probability of being a center
            for (int t = 0; t < P_S.size(); t++) {

                currentSumProbability += ((WP.get(t)) * arrayDistances.get(t)) / denom;

                if (previousSumProbability <= randomX && randomX <= currentSumProbability) {

                    closestCenters.set(t, P_S.get(t));
                    S.add(P_S.remove(t)); //remove the point from P-S and add it to S
                    arrayDistances.remove(t); //remove the selected point from P-S
                    WPcenters.add(WP.remove(t));
                    closestCenters.remove(t);
                    break;

                } else {
                    previousSumProbability = currentSumProbability;
                }
            }

        }

        // Add the points of S to the end of PS, the weights of S at the end of WP and the points of S at the end of closestCenters.
        // In this way an index corresponds to the same point in the three vectors.
        P_S.addAll(S);
        WP.addAll(WPcenters);
        closestCenters.addAll(S);

        ArrayList<Vector> S2 = S;

        //Iterate for value variable iter
        for (int j = 0; j < iter; j++) {

            S2 = lloyds(P_S, closestCenters, S2, WP); // Compute the new centers with lloyd's algorithm
            closestCenters = compute_closestCenters(P_S, S2);

        }

        return S2;

    }

    //  This method compute the array containing the closest center of each point
    private static ArrayList<Vector> compute_closestCenters(ArrayList<Vector> P_S, ArrayList<Vector> S) {
        ArrayList<Vector> closestCenters = new ArrayList<>();
        Double currentDist;

        int indS=0;
            for (int i=0; i< P_S.size(); i++) {
                double minDist = Double.POSITIVE_INFINITY;

                for (int j=0; j< S.size(); j++) {
                    currentDist = Math.sqrt(Vectors.sqdist(P_S.get(i), S.get(j)));
                    if (currentDist < minDist) {
                        minDist = currentDist;
                        indS=j;
                    }
                }
                closestCenters.add(S.get(indS));
            }
            return closestCenters;
        }


        private static ArrayList<Vector> lloyds(ArrayList < Vector > P_S, ArrayList < Vector > clusterIndex, ArrayList < Vector > S, ArrayList < Long > WP){
            double[] arrayOfZeros = new double[P_S.get(0).size()];
            Vector sum = Vectors.dense(arrayOfZeros); //create vector for the sum
            Double denom = 0.0;

            ArrayList<Vector> centers = new ArrayList<>();
            for (int i = 0; i < S.size(); i++) {
                denom = 0.0;
                BLAS.scal(0.0, sum); //set to zero the vector sum

                for (int j = 0; j < P_S.size(); j++) {
                    if (clusterIndex.get(j) == S.get(i)) {

                        BLAS.axpy(WP.get(j), P_S.get(j), sum);//sum all weighted points of single cluster
                        denom += WP.get(j);
                    }

                }
                BLAS.scal((1.0 / denom), sum); //Compute the formula: (1/sum_{p in C} w(p)) * sum_{p in C} p*w(p)
                centers.add(sum.copy()); //Add the new center to array of centers

            }
            return centers;
        }

    }







