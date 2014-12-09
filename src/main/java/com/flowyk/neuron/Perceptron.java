package com.flowyk.neuron;

import com.flowyk.neuron.messenger.DetailedActivationResult;
import com.flowyk.neuron.messenger.TrainingInput;
import com.flowyk.neuron.messenger.TrainingResult;
import com.flowyk.neuron.transferfunction.TransferFunction;
import com.sun.istack.internal.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class Perceptron extends McCullochPittsNeuron {
    private static final Logger LOG = LoggerFactory.getLogger(Perceptron.class);

    public Perceptron(@NotNull List<BigDecimal> weights, @NotNull TransferFunction transferFunction, BigDecimal learningRate) {
        super(weights, transferFunction, learningRate);
    }

    @Override
    public TrainingResult train(TrainingInput input) {
        DetailedActivationResult activationResult = activate(input);

        BigDecimal output = activationResult.getOutput();
        LOG.debug("Output: {}, desired: {}", output, input.getDesiredOutput());

        BigDecimal error = input.getDesiredOutput().subtract(output);
        TrainingResult result = new TrainingResult(activationResult.getSensorOutputs(), output, error);
        if (error.compareTo(BigDecimal.ZERO) != 0) {
            trainWeights(input, result);
        }
        return result;
    }

    private void trainWeights(TrainingInput input, TrainingResult output) {
        List<BigDecimal> inputValues = input.getInput();
        List<BigDecimal> newWeights = new ArrayList<>();

        BigDecimal desiredOutput = input.getDesiredOutput();
        BigDecimal realOutput = output.getOutput();
        BigDecimal error = desiredOutput.subtract(realOutput);
        for (int i = 0; i < weights.size() && i < inputValues.size(); i++) {
            BigDecimal inputForIndex = inputValues.get(i);
            BigDecimal correction = learningRate.multiply(inputForIndex).multiply(error);
            BigDecimal newWeight = correction.add(weights.get(i));
            newWeights.add(newWeight);
            LOG.debug("Correction: {}, newWeight: {}", correction, newWeight);
        }
        this.weights = Collections.unmodifiableList(newWeights);
    }
}
