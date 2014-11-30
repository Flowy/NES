package com.flowyk.perceptron;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class StepFunction implements TransferFunction {

    private final double threshold;

    public StepFunction(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public double transfer(Number sum) {
        if (sum.doubleValue() >= threshold) {
            return 1d;
        } else {
            return 0d;
        }
    }
}
