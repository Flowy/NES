package com.flowyk.neuralnetworks.trainers;

import com.flowyk.neuralnetworks.messenger.ActivationInput;
import com.flowyk.neuralnetworks.messenger.NeuronOutput;
import com.flowyk.neuralnetworks.messenger.TrainingInput;
import com.flowyk.neuralnetworks.neuron.McCullochPittsNeuron;
import com.flowyk.neuralnetworks.neuron.Neuron;
import com.flowyk.neuralnetworks.transferfunction.SignFunction;
import com.flowyk.neuralnetworks.transferfunction.TransferFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 13. 12. 2014.
 */
public class MadalineNetwork {
    private static final Logger LOG = LoggerFactory.getLogger(MadalineNetwork.class);

    private TransferFunction transferFunction = new SignFunction();
    private Neuron Z1 = new McCullochPittsNeuron(Arrays.asList(BigDecimal.ONE, BigDecimal.ZERO), transferFunction, BigDecimal.valueOf(1d));
    private Neuron Z2 = new McCullochPittsNeuron(Arrays.asList(BigDecimal.ZERO, BigDecimal.ONE), transferFunction, BigDecimal.valueOf(1d));
    private Neuron Y = new McCullochPittsNeuron(Arrays.asList(BigDecimal.valueOf(0.5d), BigDecimal.valueOf(0.5d)), transferFunction, BigDecimal.valueOf(0.5d));


    public void trainAll(List<TrainingInput> inputs) {
        List<NeuronOutput> z1LastOutputs;
        List<NeuronOutput> z1ActualOutputs = Collections.emptyList();
        List<NeuronOutput> z2LastOutputs;
        List<NeuronOutput> z2ActualOutputs = Collections.emptyList();
        int iterations = 0;
        do {
            LOG.debug("Training iteration {}...", iterations++);
            z1LastOutputs = z1ActualOutputs;
            z2LastOutputs = z2ActualOutputs;
            z1ActualOutputs = new ArrayList<>(inputs.size());
            z2ActualOutputs = new ArrayList<>(inputs.size());
            for (TrainingInput input : inputs) {
                NeuronOutput z1Output = Z1.activate(input);
                NeuronOutput z2Output = Z2.activate(input);
                ActivationInput yInput = new ActivationInput(Arrays.asList(z1Output.getOutput(), z2Output.getOutput()), BigDecimal.ONE);
                NeuronOutput yOutput = Y.activate(yInput);
                LOG.debug("Inputs: {}\nOutputs before training: Z1: {}, Z2: {}, Y: {}", input, z1Output, z2Output, yOutput);
                if (yOutput.getOutput().compareTo(input.getDesiredOutput()) != 0) {
                    BigDecimal z1Sum = BigDecimal.ZERO;
                    for (BigDecimal z1Out : z1Output.getSensorOutputs()) {
                        z1Sum = z1Sum.add(z1Out);
                    }
                    Z1.train(input, input.getDesiredOutput().subtract(z1Sum));

                    BigDecimal z2Sum = BigDecimal.ZERO;
                    for (BigDecimal z2Out : z2Output.getSensorOutputs()) {
                        z2Sum = z2Sum.add(z2Out);
                    }
                    Z2.train(input, input.getDesiredOutput().subtract(z2Sum));

                    LOG.debug("Output after training: {}", activate(input));
                }
                z1ActualOutputs.add(z1Output);
                z2ActualOutputs.add(z2Output);
            }
        } while (!z1LastOutputs.equals(z1ActualOutputs) && !z2LastOutputs.equals(z2ActualOutputs));
    }

    public NeuronOutput activate(ActivationInput input) {
        NeuronOutput z1Output = Z1.activate(input);
        NeuronOutput z2Output = Z2.activate(input);
        ActivationInput yInput = new ActivationInput(Arrays.asList(z1Output.getOutput(), z2Output.getOutput()), BigDecimal.ONE);
        return Y.activate(yInput);
    }
}
