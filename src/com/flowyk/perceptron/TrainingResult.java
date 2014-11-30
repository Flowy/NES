package com.flowyk.perceptron;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class TrainingResult extends ActivationResult {

    private double error;

    public TrainingResult(double output, double error) {
        super(output);
        this.error = error;
    }

    public double getError() {
        return error;
    }
}
