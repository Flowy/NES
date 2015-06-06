package com.flowyk.neuralnetworks.neuron;

import com.flowyk.neuralnetworks.messenger.ActivationInput;
import com.flowyk.neuralnetworks.messenger.NeuronOutput;
import com.flowyk.neuralnetworks.transferfunction.TransferFunction;
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
public class McCullochPittsNeuron implements Neuron {
    private static final Logger LOG = LoggerFactory.getLogger(McCullochPittsNeuron.class);

    protected List<BigDecimal> weights = new ArrayList<>();
    protected BigDecimal biasWeight;
    protected final BigDecimal learningRate = BigDecimal.valueOf(0.2d);
    protected final TransferFunction transferFunction;

    public McCullochPittsNeuron(@NotNull List<BigDecimal> sensorWeights, @NotNull TransferFunction transferFunction, @NotNull BigDecimal bias) {
        LOG.debug("Creating new neuron...");
        this.weights = sensorWeights;
        this.transferFunction = transferFunction;
        this.biasWeight = bias;
        LOG.debug("Neuron created: {}", this);
    }

    @Override
    public void train(ActivationInput input, BigDecimal error) {
        List<BigDecimal> inputValues = input.getInput();
        List<BigDecimal> newWeights = new ArrayList<>();
        for (int i = 0; i < weights.size(); i++) {
            BigDecimal inputForSensor = inputValues.get(i);
            BigDecimal correction = learningRate.multiply(inputForSensor).multiply(error).stripTrailingZeros();
            BigDecimal newWeight = correction.add(weights.get(i));
            newWeights.add(newWeight);
        }
        this.weights = Collections.unmodifiableList(newWeights);

        //biasWeight train
        biasWeight = biasWeight.add(learningRate.multiply(error));
    }

    @Override
    public NeuronOutput activate(ActivationInput input) {
        List<BigDecimal> inputValues = input.getInput();
        int numOfInputs = inputValues.size();
        if (numOfInputs != weights.size()) {
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

        BigDecimal output = sum.add(input.getBias().multiply(biasWeight));
        output = transferFunction.transfer(output);

        return new NeuronOutput(sensorOutputs, output);
    }

    @Override
    public List<BigDecimal> getWeights() {
        return Collections.unmodifiableList(weights);
    }

    @Override
    public String toString() {
        return "McCullochPittsNeuron{" +
                "weights=" + weights +
                ", biasWeight=" + biasWeight +
                '}';
    }
}
