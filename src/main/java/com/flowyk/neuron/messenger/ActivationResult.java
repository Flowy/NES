package com.flowyk.neuron.messenger;

import java.math.BigDecimal;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class ActivationResult {

    private BigDecimal output;
    public ActivationResult(BigDecimal output) {
        this.output = output;
    }

    public BigDecimal getOutput() {
        return output;
    }
}
