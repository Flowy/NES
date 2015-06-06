package com.flowyk.neuralnetworks;

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


public class MadalineXORTest {
    private static final Logger LOG = LoggerFactory.getLogger(MadalineXORTest.class);

    private static MadalineNetwork testedNeuron = new MadalineNetwork();

    private static TrainingInput getTrainingInput1() {
        return new TrainingInput(Arrays.asList(ONE, ONE), ONE, ONE.negate());
    }
    private static TrainingInput getTrainingInput2() {
        return new TrainingInput(Arrays.asList(ONE, ONE.negate()), ONE, ONE);
    }
    private static TrainingInput getTrainingInput3() {
        return new TrainingInput(Arrays.asList(ONE.negate(), ONE), ONE, ONE);
    }
    private static TrainingInput getTrainingInput4() {
        return new TrainingInput(Arrays.asList(ONE.negate(), ONE.negate()), ONE, ONE.negate());
    }
    @BeforeClass
    public static void setUp() throws Exception {

        List<TrainingInput> wikiExampleNANDSet = new ArrayList<>();
        wikiExampleNANDSet.add(getTrainingInput1());
        wikiExampleNANDSet.add(getTrainingInput2());
        wikiExampleNANDSet.add(getTrainingInput3());
        wikiExampleNANDSet.add(getTrainingInput4());

        testedNeuron.trainAll(wikiExampleNANDSet);
    }

    @Test
    public void testActivateXOR1() throws Exception {
        TrainingInput input = getTrainingInput1();
        NeuronOutput result = testedNeuron.activate(input);
        assertEquals(input.getDesiredOutput(), result.getOutput());
    }

    @Test
    public void testActivateXOR2() throws Exception {
        TrainingInput input = getTrainingInput2();
        NeuronOutput result = testedNeuron.activate(input);
        assertEquals(input.getDesiredOutput(), result.getOutput());
    }

    @Test
    public void testActivateXOR3() throws Exception {
        TrainingInput input = getTrainingInput3();
        NeuronOutput result = testedNeuron.activate(input);
        assertEquals(input.getDesiredOutput(), result.getOutput());
    }

    @Test
    public void testActivateXOR4() throws Exception {
        TrainingInput input = getTrainingInput4();
        NeuronOutput result = testedNeuron.activate(input);
        assertEquals(input.getDesiredOutput(), result.getOutput());
    }
}