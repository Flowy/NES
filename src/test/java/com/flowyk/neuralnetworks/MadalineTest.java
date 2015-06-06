package com.flowyk.neuralnetworks;

import com.flowyk.neuralnetworks.messenger.ActivationInput;
import com.flowyk.neuralnetworks.messenger.NeuronOutput;
import com.flowyk.neuralnetworks.messenger.TrainingInput;
import com.flowyk.neuralnetworks.trainers.MadalineNetwork;
import org.junit.BeforeClass;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.math.BigDecimal.ONE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class MadalineTest {
    private static final Logger LOG = LoggerFactory.getLogger(MadalineTest.class);

    private static MadalineNetwork testedNeuron = new MadalineNetwork();

    @BeforeClass
    public static void setUp() throws Exception {

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE.negate(), ONE.negate()), ONE, ONE.negate()));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE.negate(), ONE), ONE, ONE.negate()));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE.negate()), ONE, ONE.negate()));
        wikiExampleNANDSet.add(new TrainingInput(Arrays.asList(ONE, ONE), ONE, ONE));

        testedNeuron.trainAll(wikiExampleNANDSet);
    }

    @Test
    public void testActivateWikiNAND1() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE.negate(), ONE.negate()), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.negate().compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND2() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE.negate(), ONE), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.negate().compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND3() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE.negate()), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertTrue(ONE.negate().compareTo(result.getOutput()) == 0);
    }

    @Test
    public void testActivateWikiNAND4() throws Exception {
        ActivationInput nandInput = new ActivationInput(Arrays.asList(ONE, ONE), ONE);
        NeuronOutput result = testedNeuron.activate(nandInput);
        assertEquals(ONE, result.getOutput());
    }
}