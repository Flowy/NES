package com.flowyk.neuron.transferfunction;

import java.math.BigDecimal;

/**
 * Created by Lukas on 6. 12. 2014.
 */
public class SaturatedLinearFunction implements TransferFunction {
    @Override
    public BigDecimal transfer(BigDecimal sum) {
        if (sum.compareTo(BigDecimal.ONE) >= 0) {
            return BigDecimal.ONE;
        } else if (sum.compareTo(BigDecimal.ZERO) < 0) {
            return BigDecimal.ZERO;
        } else {
            return sum;
        }
    }
}
