package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationResult;
import com.flowyk.neuron.messenger.TrainingInput;
import com.flowyk.neuron.messenger.TrainingResult;
import com.flowyk.neuron.transferfunction.TransferFunction;
import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class Adaline extends McCullochPittsNeuron {

    public Adaline(@NotNull List<BigDecimal> weights, @NotNull TransferFunction transferFunction, BigDecimal learningRate) {
        super(weights, transferFunction, learningRate);
    }

    @Override
    public TrainingResult train(TrainingInput input) {
        ActivationResult activationResult = activate(input);

        double output = activationResult.getOutput();

        double error = input.getDesiredOutput().doubleValue() - output;
        if (error != 0) {
            trainWeights(input, activationResult, error);
        }
        return new TrainingResult(activationResult.getSensorOutputs(), output, error);
    }

    private void trainWeights(TrainingInput input, ActivationResult output, double error) {
    }
}
