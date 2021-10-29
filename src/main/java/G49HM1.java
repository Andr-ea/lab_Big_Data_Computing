import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;


public class G49HM1 {

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }
        //Point 1:
        // Read a list of numbers from the program options
        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s = new Scanner(new File(args[0]));
        while (s.hasNext()) {
            lNumbers.add(Double.parseDouble(s.next()));
        }
        s.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true).setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);

        //Point 2: Computes and prints the maximum value in dNumbers.

        //using the reduce method of the RDD interface;
        double max1 = dNumbers.reduce((x, y) -> {
            if (x >= y)
                 return  x;
            else return y;
        });

        System.out.println("The maximum value calculated using the reduce method: " + max1);

        //using the max method of the RDD interface
        double max2 = dNumbers.max(new MComparator());
        System.out.println("The maximum value calculated using the max method of the RDD interface:  " + max2);


        //Point 3: Creates a new RDD dNormalized containing the values of dNumbers normalized in [0,1].

        //To calculate the dNormalized we use the following formula: (x - min) / (max - min)
        //In this way we represent the minimum value as 0 and the maximum value as 1
        double min = dNumbers.min(new MComparator());
        JavaRDD<Double> dNormalized = dNumbers.map((x) -> (x - min) / (max1 - min) );

        //Point 4: Computes and prints a statistics of your choice on dNormalized.
        //Make sure that you use at least one new method provided by the RDD interface.

        //We've choosen to calculate the variance as a statistic of our RDD dNormalized, using the method count provided by its interface.

        //First we've calculated the mean of the normalized values and then we've used it to calculate the variance.
        double mean = (dNormalized.reduce((x, y) -> x + y)) / dNormalized.count();

        double variance= dNormalized.map(x-> Math.pow((x-mean),2) ).reduce((x,y)-> x+y )/dNormalized.count();

        System.out.println("The variance of dNormalized is " + variance);


    }

    public static class MComparator implements Serializable, Comparator<Double> {

        public int compare(Double a, Double b) {
            if (a < b) return -1;
            else if (a > b) return 1;
            return 0;
        }
    }
}




