package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationInput;
import com.flowyk.neuron.messenger.ActivationResult;
import com.flowyk.neuron.messenger.TrainingInput;
import com.flowyk.neuron.transferfunction.BipolarnaFunkcia;
import com.flowyk.neuron.transferfunction.TransferFunction;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class AdalineTest {
    private static final Logger LOG = LoggerFactory.getLogger(AdalineTest.class);

    private static McCullochPittsNeuron testedNeuron;

    @BeforeClass
    public static void setUp() throws Exception {
        List<BigDecimal> initialWeights = Arrays.asList(BigDecimal.ZERO, BigDecimal.ZERO, BigDecimal.ZERO);
        TransferFunction aktivacnaFunkcia = new BipolarnaFunkcia(0.5d);
        testedNeuron = new Adaline(initialWeights, aktivacnaFunkcia, BigDecimal.valueOf(0.1d));

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 0d, 0d), 1d));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 0d, 1d), 1d));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 1d, 0d), 1d));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(1d, 1d, 1d), 0d));

        testedNeuron.learnBySet(wikiExampleNANDSet);
    }

    @Test
    public void testActivateWikiNAND1() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 0d, 0d));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertEquals(1d, result.getOutput(), 0.01d);
    }

    @Test
    public void testActivateWikiNAND2() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 0d, 1d));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertEquals(1d, result.getOutput(), 0.01d);
    }

    @Test
    public void testActivateWikiNAND3() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 1d, 0d));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertEquals(1d, result.getOutput(), 0.01d);
    }

    @Test
    public void testActivateWikiNAND4() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(1d, 1d, 1d));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertEquals(0d, result.getOutput(), 0.01d);
    }
}