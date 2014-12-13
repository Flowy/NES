package com.flowyk.neuralnetworks;

import com.flowyk.neuralnetworks.messenger.ActivationInput;
import com.flowyk.neuralnetworks.messenger.NeuronOutput;
import com.flowyk.neuralnetworks.messenger.TrainingInput;
import com.flowyk.neuralnetworks.neuron.McCullochPittsNeuron;
import com.flowyk.neuralnetworks.neuron.Neuron;
import com.flowyk.neuralnetworks.trainers.NeuronTrainer;
import com.flowyk.neuralnetworks.trainers.PerceptronTrainer;
import com.flowyk.neuralnetworks.transferfunction.SignFunction;
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
import static org.junit.Assert.assertTrue;

public class PerceptronTest2 {
    private static final Logger LOG = LoggerFactory.getLogger(PerceptronTest2.class);

    private static Neuron testedNeuron;

    @BeforeClass
    public static void setUp() throws Exception {
        TransferFunction aktivacnaFunkcia = new SignFunction();

        testedNeuron = new McCullochPittsNeuron(2, aktivacnaFunkcia, BigDecimal.valueOf(0.5d));
        NeuronTrainer neuronTrainer = new PerceptronTrainer(testedNeuron);

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE.negate(), ONE.negate()), ONE, ONE));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE.negate(), ONE), ONE, ONE));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE.negate()), ONE, ONE));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE), ONE, ONE.negate()));

        neuronTrainer.trainAll(wikiExampleNANDSet);
    }

    @Test
    public void testActivateWikiNAND1() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE.negate(), ONE.negate()), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND2() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE.negate(), ONE), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND3() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE.negate()), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND4() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.negate().compareTo(result.getOutput()) == 0);
    }
}