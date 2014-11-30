package com.flowyk.neuron.messenger;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class TrainingResult extends ActivationResult {

    private double error;

    public TrainingResult(List<BigDecimal> sensorOutputs, double output, double error) {
        super(sensorOutputs, output);
        this.error = error;
    }

    public double getError() {
        return error;
    }
}
