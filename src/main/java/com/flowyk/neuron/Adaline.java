package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationResult;
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
public class Adaline extends McCullochPittsNeuron {
    private static final Logger LOG = LoggerFactory.getLogger(Adaline.class);

    public Adaline(@NotNull List<BigDecimal> weights, @NotNull TransferFunction transferFunction, BigDecimal learningRate) {
        super(weights, transferFunction, learningRate);
    }

    @Override
    public TrainingResult train(TrainingInput input) {
        ActivationResult activationResult = activate(input);

        double output = activationResult.getOutput();
        LOG.debug("Output: {}, desired: {}", output, input.getDesiredOutput().doubleValue());

        double error = input.getDesiredOutput().doubleValue() - output;
        TrainingResult result = new TrainingResult(activationResult.getSensorOutputs(), output, error);
        double theta = 0.001d;
        if (error > theta || error < -theta) {
            trainWeights(input, activationResult);
        }
        return result;
    }

    private void trainWeights(TrainingInput input, ActivationResult output) {
        List<Number> inputValues = input.getInput();
        List<BigDecimal> newWeights = new ArrayList<>();

        BigDecimal desiredOutput = BigDecimal.valueOf(input.getDesiredOutput().doubleValue());

        for (int i = 0; i < weights.size() && 1 < inputValues.size(); i++) {
            BigDecimal inputForIndex = BigDecimal.valueOf(inputValues.get(i).doubleValue());
            BigDecimal realOutput = BigDecimal.valueOf(output.getSensorOutputs().get(i).doubleValue());
            BigDecimal error = desiredOutput.subtract(realOutput);
            BigDecimal correction = learningRate.multiply(inputForIndex).multiply(error);
            BigDecimal newWeight = correction.add(weights.get(i));
            newWeights.add(newWeight);
            LOG.debug("Sensor: {}, input: {}, Correction: {}, newWeight: {}", i, inputForIndex, correction, newWeight);
        }
        this.weights = Collections.unmodifiableList(newWeights);
    }
}
