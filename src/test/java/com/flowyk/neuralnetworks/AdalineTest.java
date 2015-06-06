package com.flowyk.neuralnetworks;

import com.flowyk.neuralnetworks.messenger.ActivationInput;
import com.flowyk.neuralnetworks.messenger.NeuronOutput;
import com.flowyk.neuralnetworks.messenger.TrainingInput;
import com.flowyk.neuralnetworks.neuron.McCullochPittsNeuron;
import com.flowyk.neuralnetworks.trainers.NeuralNetworkTrainer;
import com.flowyk.neuralnetworks.trainers.PerceptronTrainer;
import com.flowyk.neuralnetworks.transferfunction.StepFunction;
import com.flowyk.neuralnetworks.transferfunction.TransferFunction;
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
        TransferFunction aktivacnaFunkcia = new StepFunction(BigDecimal.valueOf(0.5d));

        testedNeuron = new McCullochPittsNeuron(Arrays.asList(ZERO, ZERO), aktivacnaFunkcia, ZERO);
        NeuralNetworkTrainer neuronTrainer = new PerceptronTrainer(testedNeuron);

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ZERO, ZERO), ONE, ZERO));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ZERO, ONE), ONE, ZERO));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ZERO), ONE, ZERO));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE), ONE, ONE));

        neuronTrainer.trainAll(wikiExampleNANDSet);
    }

    @Test
    public void testActivateWikiNAND1() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ZERO, ZERO), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ZERO.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND2() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ZERO, ONE), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ZERO.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND3() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ZERO), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ZERO.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND4() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }
}