package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationInput;
import com.flowyk.neuron.messenger.DetailedActivationResult;
import com.flowyk.neuron.messenger.TrainingInput;
import com.flowyk.neuron.messenger.TrainingResult;
import com.flowyk.neuron.transferfunction.TransferFunction;
import com.sun.istack.internal.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public abstract class McCullochPittsNeuron {
    private static final Logger LOG = LoggerFactory.getLogger(McCullochPittsNeuron.class);

    protected List<BigDecimal> weights;
    protected final BigDecimal learningRate;
    protected final TransferFunction transferFunction;

    public McCullochPittsNeuron(@NotNull List<BigDecimal> weights, @NotNull TransferFunction transferFunction, @NotNull BigDecimal learningRate) {
        LOG.debug("Creating new neuron...");
        this.weights = weights;
        this.transferFunction = transferFunction;
        this.learningRate = learningRate;
    }

    public abstract TrainingResult train(TrainingInput input);

    public DetailedActivationResult activate(ActivationInput input) {
        List<BigDecimal> inputValues = input.getInput();
        List<BigDecimal> sensorOutputs = new ArrayList<>();
        BigDecimal sum = BigDecimal.ZERO;
        for (int i = 0; i < weights.size() && i < inputValues.size(); i++) {
            BigDecimal inputValue = inputValues.get(i);
            BigDecimal weight = weights.get(i);
            BigDecimal sensorOutput = weight.multiply(inputValue);
            sensorOutputs.add(sensorOutput);
            sum = sum.add(sensorOutput);
        }

        BigDecimal output = transferFunction.transfer(sum);

        return new DetailedActivationResult(sensorOutputs, output);
    }

    public void learnBySet(List<TrainingInput> inputs) {
        List<TrainingResult> outputs;
        int iterations = 0;
        LOG.debug("Learning started: {}", this);
        do {
            LOG.debug("Learn cycle starting...");
            outputs = trainSet(inputs);
            LOG.debug("Learn cycle finished: {} ", this);
            iterations++;
        } while (checkError(outputs));
        LOG.info("Learning done after {} iterations: {}", iterations, this);
    }

    public List<TrainingResult> trainSet(List<TrainingInput> inputs) {
        List<TrainingResult> results = new ArrayList<>();
        for (TrainingInput input: inputs) {
            results.add(train(input));
        }
        return results;
    }

    private boolean checkError(List<TrainingResult> outputs) {
        boolean error = false;
        double theta = 0.0001d;
        for (TrainingResult result: outputs) {
            LOG.debug("Error per result: {}", result.getError());
            if (result.getError().compareTo(BigDecimal.ZERO) != 0) {
                error = true;
                if (!LOG.isDebugEnabled()) {
                    break;
                }
            }
        }
        return error;
    }

    @Override
    public String toString() {
        return "McCullochPittsNeuron{" +
                "weights=" + weights +
                '}';
    }
}
