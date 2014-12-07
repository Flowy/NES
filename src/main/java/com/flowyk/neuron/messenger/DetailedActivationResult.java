package com.flowyk.neuron.messenger;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 7. 12. 2014.
 */
public class DetailedActivationResult extends ActivationResult {


    private List<BigDecimal> sensorOutputs;

    public DetailedActivationResult(List<BigDecimal> sensorOutputs, double output) {
        super(output);
        this.sensorOutputs = Collections.unmodifiableList(sensorOutputs);
    }

    public List<BigDecimal> getSensorOutputs() {
        return sensorOutputs;
    }
}
