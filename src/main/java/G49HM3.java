import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class G49HM3 {

    public static void main(String[] args) throws IOException {

        ArrayList<Vector> P = VectorInput.readVectorsSeq("covtype1K.data.txt");
        ArrayList<Long> WP = createWeights(P.size(), 1L);

        ArrayList<Vector> centers = kmeansPP(P, WP, 25, 3);
        double average = kmeansObj(P, centers);

        System.out.printf("The average distance of points of P from C is: " + average);

    }

    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> WP, Integer k, Integer iter) {
        ArrayList<Vector> S = new ArrayList<>();    // create set of centers
        ArrayList<Vector> P_S = new ArrayList<>();
        P_S.addAll(P); //create P-S

        //extract a random point with uniform probability
        int bound = (int) (Math.random() * P.size());
        S.add(P_S.remove(bound)); //select the first center randomly and add it to set of centers S and we remove it from P-S

        //iterate for number of clusters
        for (int i = 1; i < k; i++) {

            //Initialize the usefull variable to compute minimum distance
            double dist = 0;
            double denom = 0;

            HashMap<Vector, Double> arrayDistances = new HashMap<>();

            for (Vector p : P_S) { //Iterate for each point of P-S

                //Iterate for each center
                Vector lastcenter = S.get(S.size() - 1);

                dist = Math.sqrt(Vectors.sqdist(p, lastcenter));  //Compute distance between p and c
                if (!arrayDistances.containsKey(p)) {
                    arrayDistances.put(p, Double.POSITIVE_INFINITY);
                }
                if (dist < arrayDistances.get(p)) {
                    arrayDistances.replace(p, dist);
                }


                denom += (WP.get(P.indexOf(p)) * arrayDistances.get(p)); // Compute the denominator of probability
            }

            double randomX = Math.random();    // Compute random number x between 0 and 1
            double currentSumProbability = 0;  // Variable used to compute the sum of probability including the last point
            double previousSumProbability = 0; // Variable used to compute the sum of probability excluding the last point

            Vector newCenters;
            int indMin = 1;

            //Iterate P-S and select the point with higher probability of being a center
            for (Vector t : arrayDistances.keySet()) {

                currentSumProbability += ((WP.get(P.indexOf(t)) * arrayDistances.get(t)) / denom);

                if (previousSumProbability <= randomX && randomX <= currentSumProbability) {
                    newCenters = t.copy();
                    S.add(P_S.remove(P_S.indexOf(newCenters))); //remove the point from P-S and add it to S
                    arrayDistances.remove(t); //remove the selected point from P-S
                    break;

                } else {
                    previousSumProbability = currentSumProbability;
                }
            }
        }

        ArrayList<Vector> S2 = S; //List of k centers compute with kmeans++
        Map<Vector, ArrayList<Vector>> cluster = new HashMap<>();

        //Iterate for value variable iter
        for (int j = 0; j < iter; j++) {
            cluster = partition(P, S2);// This map for each cluster store the center and the index of the points

            S2 = lloyds(P, cluster, WP); // Compute the new centers with lloyd's algorithm

        }
        cluster = partition(P, S2);

        return S2;

    }

    private static ArrayList<Vector> lloyds(ArrayList<Vector> P, Map<Vector, ArrayList<Vector>> cluster, ArrayList<Long> WP) {

        ArrayList<Vector> centers = new ArrayList<>();

        double[] arrayOfZeros = new double[P.get(0).size()];
        Vector sum = Vectors.dense(arrayOfZeros); //create vector for the sum

        Double denom = 0.0;

        for (ArrayList<Vector> cl : cluster.values()) {
            denom = 0.0;
            BLAS.scal(0.0, sum); //azzero il vettore somma

            for (Vector vec : cl) {
                BLAS.axpy(WP.get(P.indexOf(vec)), vec, sum);//sum all weighted points of single cluster
                denom += WP.get(P.indexOf(vec));
            }

            BLAS.scal((1.0 / denom), sum); //Compute the formula: (1/sum_{p in C} w(p)) * sum_{p in C} p*w(p)
            centers.add(sum.copy()); //Sdd the new center to array of centers
        }
        return centers; //Return the list of new centers

    }

    //Method used to compute a weighted array
    public static ArrayList<Long> createWeights(Integer numb, Long value) {
        ArrayList<Long> w = new ArrayList<>();
        for (int i = 0; i < numb; i++) {
            w.add(value);
        }
        return w;
    }

    //Method that assign each point to its closest center
    public static Map<Vector, ArrayList<Vector>> partition(ArrayList<Vector> P, ArrayList<Vector> S) {
        double dist = 0;
        double dmin;

        Vector closestCenter = P.get(0); // Initialize closest center with a random vector
        Map<Vector, ArrayList<Vector>> cluster = new HashMap<>();//Map that represents the cluster(key-center of cluster = value-index of points of cluster )

        //Iterate all points pi of P
        for (Vector pi : P) {

            dmin = Double.POSITIVE_INFINITY;

            //Iterate the centers and compute the closest to pi
            for (Vector c : S) {
                dist = Math.sqrt(Vectors.sqdist(pi, c));

                if (dist < dmin) {
                    dmin = dist;
                    closestCenter = c.copy(); //Assign the closest center of pi to the vector closestCenter
                }
            }
            //Add point pi to list of its closest center
            if (cluster.keySet().contains(closestCenter)) {
                cluster.get(closestCenter).add(pi);
            } else {
                cluster.put(closestCenter, new ArrayList<>());
                cluster.get(closestCenter).add(pi);
            }
        }
        return cluster;

    }

    public static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C) {
        // Compute the mean of the distances between each point and its closest center
        double currentDist;
        double sum = 0;
        for (Vector p : P) {
            double minDist = Double.POSITIVE_INFINITY;
            for (Vector c : C) {
                currentDist = Math.sqrt(Vectors.sqdist(p, c));
                if (currentDist < minDist) {
                    minDist = currentDist;
                }
            }
            // Sum the min distance
            sum += minDist;
        }
        // Return the mean
        return sum / P.size();
    }

}