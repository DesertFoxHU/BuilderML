package me.desertfox.testml;

import me.desertfox.testml.utils.DownloaderUtility;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Test {

    public static String dataLocalPath;

    private static void log(String msg){
        System.out.println(msg);
    }

    private static List<String> processScript(ProcessBuilder builder){
        try {
            Process process = builder.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            List<String> lines = new ArrayList<>();
            String line;
            while((line = reader.readLine()) != null){
                lines.add(line);
            }

            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                if (errorLine.contains("error") || errorLine.contains("Exception")) {
                    log("Error: " + errorLine);
                } else {
                    log("Info: " + errorLine);
                }
            }

            process.waitFor();
            return lines;
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws Exception {
        dataLocalPath = DownloaderUtility.NLPDATA.Download();

        String filePath = new File(dataLocalPath, "raw_sentences.txt").getAbsolutePath();
        log("Load & Vectorize Sentences...");

        SentenceIterator iter = new BasicLineIterator(filePath);
        TokenizerFactory t = new DefaultTokenizerFactory();

        Scanner scanner = new Scanner(System.in);
        log("Tell me a sentence:");
        log("");
        String in = scanner.nextLine();
        scanner.close();

        in = in.toLowerCase();

        //TODO: Prone to exception
        in = processScript(new ProcessBuilder("python", "nltk_stopword.py", in)).get(0);

        log(in);
        t.setTokenPreProcessor(new CommonPreprocessor());
        Tokenizer tokenizer = t.create(in);
        List<String> tokens = new ArrayList<>();
        while(tokenizer.hasMoreTokens()){
            tokens.add(tokenizer.nextToken());
        }

        log("Building model...");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log("Fitting Word2Vec model...");
        vec.fit();

        log("Writing word vectors to text file...");

        for(String token : tokens){
            log("Token: " + token);
            if(!vec.hasWord(token)){
                log("No related word(s) found!");
                continue;
            }
            log("Closest words: " + vec.wordsNearestSum(token, 5));
        }
    }

}