package com.flowyk.neuron.transferfunction;

import java.math.BigDecimal;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class StepFunction implements TransferFunction {

    private final BigDecimal threshold;

    public StepFunction(BigDecimal threshold) {
        this.threshold = threshold;
    }

    @Override
    public BigDecimal transfer(BigDecimal sum) {
        if (sum.compareTo(threshold) >= 0) {
            return BigDecimal.ONE;
        } else {
            return BigDecimal.ZERO;
        }
    }
}
