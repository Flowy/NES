package com.flowyk.neuron.messenger;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class TrainingResult extends DetailedActivationResult {

    private BigDecimal error;

    public TrainingResult(@NotNull List<BigDecimal> sensorOutputs, @NotNull BigDecimal output, @NotNull BigDecimal error) {
        super(sensorOutputs, output);
        this.error = error;
    }

    public BigDecimal getError() {
        return error;
    }
}
