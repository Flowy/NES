package com.flowyk.neuron.messenger;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class ActivationResult {

    private List<BigDecimal> sensorOutputs;
    private double output;
    public ActivationResult(List<BigDecimal> sensorOutputs, double output) {
        this.sensorOutputs = Collections.unmodifiableList(sensorOutputs);
        this.output = output;
    }

    public List<BigDecimal> getSensorOutputs() {
        return sensorOutputs;
    }

    public double getOutput() {
        return output;
    }
}
