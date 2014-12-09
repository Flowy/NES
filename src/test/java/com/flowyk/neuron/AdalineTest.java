package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationInput;
import com.flowyk.neuron.messenger.ActivationResult;
import com.flowyk.neuron.messenger.TrainingInput;
import com.flowyk.neuron.transferfunction.StepFunction;
import com.flowyk.neuron.transferfunction.TransferFunction;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.math.BigDecimal.ONE;
import static java.math.BigDecimal.ZERO;
import static org.junit.Assert.assertTrue;


public class AdalineTest {
    private static final Logger LOG = LoggerFactory.getLogger(AdalineTest.class);

    private static McCullochPittsNeuron testedNeuron;

    @BeforeClass
    public static void setUp() throws Exception {
        List<BigDecimal> initialWeights = Arrays.asList(ZERO, ZERO, ZERO);
        TransferFunction aktivacnaFunkcia = new StepFunction(BigDecimal.valueOf(0.5d));
        testedNeuron = new Adaline(initialWeights, aktivacnaFunkcia, BigDecimal.valueOf(0.1d));

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ZERO, ZERO), ONE));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ZERO, ONE), ONE));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE, ZERO), ONE));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE, ONE), ZERO));

        testedNeuron.learnBySet(wikiExampleNANDSet);
    }

    @Test
    public void testActivateWikiNAND1() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ZERO, ZERO));
        ActivationResult result = testedNeuron.activate(nandInput);

        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND2() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ZERO, ONE));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND3() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE, ZERO));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND4() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE, ONE));
        ActivationResult result = testedNeuron.activate(nandInput);
        assertTrue(ZERO.compareTo(result.getOutput()) == 0);
    }
}