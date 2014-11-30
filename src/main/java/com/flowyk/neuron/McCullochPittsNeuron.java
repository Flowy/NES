package com.flowyk.neuron;

import com.flowyk.neuron.messenger.ActivationInput;
import com.flowyk.neuron.messenger.ActivationResult;
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
        this.weights = weights;
        this.transferFunction = transferFunction;
        this.learningRate = learningRate;
    }

    public abstract TrainingResult train(TrainingInput input);

    public ActivationResult activate(ActivationInput input) {
        List<Number> inputValues = input.getInput();
        List<BigDecimal> sensorOutputs = new ArrayList<>();
        BigDecimal sum = BigDecimal.ZERO;
        for (int i = 0; i < weights.size() && i < inputValues.size(); i++) {
            Number inputValue = inputValues.get(i);
            BigDecimal weight = weights.get(i);
            BigDecimal sensorOutput = weight.multiply(BigDecimal.valueOf(inputValue.doubleValue()));
            sensorOutputs.add(sensorOutput);
            sum = sum.add(sensorOutput);
        }

        double output = transferFunction.transfer(sum);

        return new ActivationResult(sensorOutputs, output);
    }

    public void learnBySet(List<TrainingInput> inputs) {
        List<TrainingResult> outputs;
        LOG.debug("Learning started: {}", this);
        do {
            outputs = trainSet(inputs);
            LOG.debug("Learn cycle finished: {}", this);
        } while (checkError(outputs));
        LOG.debug("Learning done: {}", this);
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
            LOG.debug("Error: {}", result.getError());
            if (result.getError() > theta || result.getError() < -theta) {

                error = true;
                break;
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
