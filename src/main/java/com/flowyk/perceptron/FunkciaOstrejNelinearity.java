package com.flowyk.perceptron;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class FunkciaOstrejNelinearity implements ActivationFunction {

    private final double threshold;

    public FunkciaOstrejNelinearity(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public double activate(Number sum) {
        if (sum.doubleValue() >= threshold) {
            return 1d;
        } else {
            return 0d;
        }
    }

    @Override
    public String toString() {
        return "FunkciaOstrejNelinearity{" +
                "threshold=" + threshold +
                '}';
    }
}
