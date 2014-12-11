package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationInput;
import com.flowyk.neuron.messenger.NeuronOutput;
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
public abstract class McCullochPittsNeuron implements Neuron {
    private static final Logger LOG = LoggerFactory.getLogger(McCullochPittsNeuron.class);

    protected List<BigDecimal> weights = new ArrayList<>();
    protected final BigDecimal learningRate;
    protected final TransferFunction transferFunction;

    public McCullochPittsNeuron(int numberOfSensors, @NotNull TransferFunction transferFunction, @NotNull BigDecimal learningRate) {
        LOG.debug("Creating new neuron...");
        for (int i = 0; i < numberOfSensors; i++) {
            weights.add(BigDecimal.ZERO);
        }
        this.learningRate = learningRate;
        this.transferFunction = transferFunction;
    }

    @Override
    public void train(ActivationInput input, BigDecimal error) {
        List<BigDecimal> inputValues = input.getInput();
        List<BigDecimal> newWeights = new ArrayList<>();
        //TODO: work with bias
        for (int i = 0; i < weights.size(); i++) {
            BigDecimal inputForSensor = inputValues.get(i);
            BigDecimal correction = learningRate.multiply(inputForSensor).multiply(error);
            BigDecimal newWeight = correction.add(weights.get(i));
            newWeights.add(newWeight);
        }
        this.weights = Collections.unmodifiableList(newWeights);
    }

    @Override
    public NeuronOutput activate(ActivationInput input) {
        List<BigDecimal> inputValues = input.getInput();
        int numOfInputs = inputValues.size();
        if (inputValues.size() != weights.size()) {
            throw new IllegalArgumentException("Input values (" + numOfInputs + ") must be as many as sensors (" + weights.size() + ")");
        }
        List<BigDecimal> sensorOutputs = new ArrayList<>(numOfInputs);
        BigDecimal sum = BigDecimal.ZERO;
        for (int i = 0; i < numOfInputs; i++) {
            BigDecimal inputValue = inputValues.get(i);
            BigDecimal weight = weights.get(i);
            BigDecimal sensorOutput = weight.multiply(inputValue);
            sensorOutputs.add(sensorOutput);
            sum = sum.add(sensorOutput);
        }

        return new NeuronOutput(sensorOutputs, transferFunction);
    }

    @Override
    public List<BigDecimal> getWeights() {
        return Collections.unmodifiableList(weights);
    }

    @Override
    public String toString() {
        return "McCullochPittsNeuron{" +
                "weights=" + weights +
                '}';
    }
}
