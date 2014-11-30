package com.flowyk.perceptron;

import com.sun.istack.internal.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class Perceptron {
    private static final Logger LOG = LoggerFactory.getLogger(Perceptron.class);

    private List<Double> weights;
    private double bias = 0;
    private final double learningRate;
    private final TransferFunction transferFunction;

    public Perceptron(@NotNull List<Double> weights, double learningRate, @NotNull TransferFunction transferFunction) {
        this.weights = Collections.unmodifiableList(weights);
        this.learningRate = learningRate;
        this.transferFunction = transferFunction;
    }

    public ActivationResult activate(ActivationInput input) {
        List<Number> inputValues = input.getInput();
        double sum = 0;
        for (int i = 0; i < weights.size() && i < inputValues.size(); i++) {
            Number inputValue = inputValues.get(i);
            Double weight = weights.get(i);
            double sensorOutput = evaluatePerSensor(weight, inputValue);
            sum += sensorOutput;
        }

        double output = transferFunction.transfer(bias + sum);

        return new ActivationResult(output);
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

    private boolean checkError(List<TrainingResult> outputs) {
        boolean error = false;
        for (TrainingResult result: outputs) {
            if (result.getError() != 0d) {
                error = true;
                break;
            }
        }
        return error;
    }

    public List<TrainingResult> trainSet(List<TrainingInput> inputs) {
        List<TrainingResult> results = new ArrayList<>();
        for (TrainingInput input: inputs) {
            results.add(train(input));
        }
        return results;
    }

    public TrainingResult train(TrainingInput sample) {
        ActivationResult activationResult = this.activate(sample);

        List<Number> inputValues = sample.getInput();
        double output = activationResult.getOutput();

        double error = sample.getDesiredOutput().doubleValue() - output;
        if (error != 0) {
            refreshWeightsByError(inputValues, error);
        }
        return new TrainingResult(output, error);
    }

    private void refreshWeightsByError(List<Number> inputValues, double error) {
        double correction = error * learningRate;
        List<Double> newWeights = new ArrayList<>();
        for (int i = 0; i < weights.size() && i < inputValues.size(); i++) {
            double newWeight = weights.get(i) + (inputValues.get(i).doubleValue() * correction);
            newWeights.add(newWeight);
        }
        this.weights = Collections.unmodifiableList(newWeights);
        bias = bias + correction;
    }

    private double evaluatePerSensor(Double weight, Number input) {
        return weight * input.doubleValue();
    }

    @Override
    public String toString() {
        return "Perceptron{" +
                "weights=" + weights +
                ", bias=" + bias +
                '}';
    }
}
