package com.flowyk;

import com.flowyk.test.perceptron.PerceptronTest;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Main {

    private static final Logger LOG = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        LOG.info("App running...");
        JUnitCore junit = new JUnitCore();
        Result result = junit.run(PerceptronTest.class);
        LOG.info("Test successful: {}", result.wasSuccessful());
        LOG.info("App finished.");
    }
}