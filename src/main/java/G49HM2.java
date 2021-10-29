import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import scala.Tuple2;
import java.io.*;
import java.util.*;


public class G49HM2 {

    public static void main(String[] args) throws IOException {

        //Configuration Spark
        SparkConf configuration =
                new SparkConf(true)
                        .setAppName("Second Homework")
                        .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(configuration);

        //Point 1. Reads the collection of documents into an RDD docs and subdivides it into K parts;
        System.out.println("Insert the value of K:");
        Scanner s = new Scanner(System.in);
        Integer k = s.nextInt();

        JavaRDD<String> docs = sc.textFile("text-sample.txt", k).cache();
        docs.count();

        //Point 2. Runs the MapReduce Word count algorithms and returns their individual running times, carefully measured.

        //START Time
        long start = System.currentTimeMillis();

        JavaPairRDD<String, Long> wordCount1 =  improvedWordCount1(docs);   //Compute the improvedWordCount1
        wordCount1.count();
        long step1 = System.currentTimeMillis();

        JavaPairRDD<String, Long> wordCount2Variant1 = improvedWordCount2Variant1(docs,k); //Compute the improvedWordCount2Variant1
        wordCount2Variant1.count();
        long step2 = System.currentTimeMillis();

        JavaPairRDD<String, Long> wordCount2Variant2 = improvedWordCount2Variant2(docs);  //Compute the improvedWordCount2Variant2
        wordCount2Variant2.count();
        // STOP Time
        long end = System.currentTimeMillis();

        // Print the running times of each variant
        System.out.println("Running time for the Improved Word count 1: " + (step1 - start) + " ms");
        System.out.println("Running time for the Improved Word count 2 variant 1: " + (step2 - step1) + " ms");
        System.out.println("Running time for Improved Word count 2 variant 2: " + (end - step2) + " ms");

        //Point 3. Prints the average length of the distinct words appearing in the documents.
        double average = computeAverageLength(docs);
        System.out.println("The average length of the distinct words appearing in the documents is: " + average);

        // Suspend the execution of the program to have time to visit the web interface (localhost:4040) of the running program
        System.out.println("Press enter to finish");
        System.in.read();

    }

    //Method that compute the average length of the distinct words appearing in the documents
    public static double computeAverageLength(JavaRDD<String> docs) {
        JavaPairRDD<String, Long> wordcount = improvedWordCount2Variant2(docs);

        //Compute the sum of the length of each distinct word in the documents
        double sumlength = 0.0;
        for (Tuple2<String, Long> i : wordcount.collect()) {
            sumlength += i._1().length();
        }
        //Compute the average length dividing the the sum computed previously with the number of distinct words in all documents
        double avlength = sumlength / wordcount.count();

        return avlength;

    }

    //Improved Word count 1 algorithm described in class the using reduceByKey
    public static JavaPairRDD<String, Long> improvedWordCount1( JavaRDD<String> docs ){

        JavaPairRDD<String, Long> wordcountpairs = docs
                // Map phase
                .flatMapToPair((document) -> {                    //For each document compute the set of intermediate pairs (word, occurrence)
                    String[] tokens = document.split(" ");

                    // In order to count efficiently the words in this document, we use a HashMap
                    HashMap<String, Long> counts = new HashMap<>();

                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    // Transform the hashmap into an array of tuples (word,occurence)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                    //Reduce phase
                }).reduceByKey((x,y) -> (x+y ));      // For each word in the intermediate pairs compute the sum of all occurences

        return wordcountpairs;

    }

    // Variant of the Improved Word Count 2 algorithm presented in class where random keys take K possible values
    public static JavaPairRDD<String, Long> improvedWordCount2Variant1( JavaRDD<String> docs, int k ){


        JavaPairRDD<String, Long> wordcountpairs = docs
                // Map phase
                .flatMapToPair((document) -> {         //For each document compute the set of intermediate pairs (word, occurrence)
                    String[] tokens = document.split(" ");

                    // In order to count efficiently the words in this document, we use a HashMap
                    HashMap<String, Long> counts = new HashMap<>();

                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    // Transform the hashmap into an array of tuples (word,occurence)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                }).groupBy( (w) -> (int)(Math.random()* k ) ) //Group the pairs by an assigned random keys in the range [0,k-1]

                .flatMapToPair((document)->{    //For each key compute the sum of the occurrences of the same word

                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (Tuple2<String, Long> i: document._2()) {
                        if(!counts.containsKey(i._1())) {
                            counts.put(i._1(),i._2());
                        }else{
                            counts.replace(i._1(),counts.get(i._1()) + i._2());
                        }
                    }
                    // Transform the hashmap into an array of tuples (word,count)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();

                }).reduceByKey((x,y) -> x+y);  //For each word in the intermediate pairs compute the sum of all occurences

        return wordcountpairs;

    }

    // Variant that does not explicitly assign random keys but exploits the subdivision of docs into K parts
    public static JavaPairRDD<String, Long> improvedWordCount2Variant2(JavaRDD<String> docs) {

        JavaPairRDD<String, Long> wordcountpairs = docs
                // Map phase
                .flatMapToPair((document) -> {                   //For each document compute the set of intermediate pairs (word, occurrences)
                    String[] tokens = document.split(" ");

                    HashMap<String, Long> counts = new HashMap<>();

                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();


                }).mapPartitionsToPair( (iteratorPartition) -> { //In each partition compute the sum of the occurrences of the same word

                    ArrayList<Tuple2<String, Long>> array = new ArrayList<>();

                    // In order to count efficiently the words in this document, we use a HashMap
                    HashMap<String, Long> counts2 = new HashMap<>();

                    while ( iteratorPartition.hasNext() ) {
                        Tuple2<String, Long> i = iteratorPartition.next();

                        if( ! counts2.containsKey(i._1() )){
                            counts2.put(i._1(),i._2());
                        }
                        else {
                            Long val = counts2.get(i._1() );
                            counts2.put(i._1(), val + i._2());
                        }
                    }

                    // Transform the hashmap into an array of tuples (word,count)
                    for (Map.Entry<String, Long> e : counts2.entrySet()) {
                        array.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }

                    return array.iterator();

                }).reduceByKey( (x,y) -> x+y); // For every word in the intermediate pairs of each partition compute the sum of all occurences

        return wordcountpairs;
    }
}