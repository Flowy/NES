package com.flowyk.perceptron;

import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class PerceptronTest {
    private static final Logger LOG = LoggerFactory.getLogger(PerceptronTest.class);

    private static Perceptron perceptron;

    @BeforeClass
    public static void setUp() throws Exception {
        List<Double> initialWeights = Arrays.asList(0d, 0d, 0d);
        FunkciaOstrejNelinearity aktivacnaFunkcia = new FunkciaOstrejNelinearity(0.5d);
        perceptron = new Perceptron(initialWeights, 0.1d, aktivacnaFunkcia);

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 0d, 0d), 1d));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 0d, 1d), 1d));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 1d, 0d), 1d));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 1d, 1d), 0d));

        perceptron.learnBySet(wikiExampleNANDSet);
    }

    @Test
    public void testActivateWikiNAND1() throws Exception {
        LOG.info("Testing perceptron for NAND1");
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 0d, 0d));
        ActivationResult result = perceptron.activate(nandInput);
        assertEquals(1d, result.getOutput(), 0.01d);
    }

    @Test
    public void testActivateWikiNAND2() throws Exception {
        LOG.info("Testing perceptron for NAND2");
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 0d, 1d));
        ActivationResult result = perceptron.activate(nandInput);
        assertEquals(1d, result.getOutput(), 0.01d);
    }

    @Test
    public void testActivateWikiNAND3() throws Exception {
        LOG.info("Testing perceptron for NAND3");
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 1d, 0d));
        ActivationResult result = perceptron.activate(nandInput);
        assertEquals(1d, result.getOutput(), 0.01d);
    }

    @Test
    public void testActivateWikiNAND4() throws Exception {
        LOG.info("Testing perceptron for NAND4");
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 1d, 1d));
        ActivationResult result = perceptron.activate(nandInput);
        assertEquals(0d, result.getOutput(), 0.01d);
    }
}