package com.flowyk.neuralnetworks.transferfunction;

import java.math.BigDecimal;

/**
 * Created by Lukas on 9. 12. 2014.
 */
public class SignFunction implements TransferFunction {
    @Override
    public BigDecimal transfer(BigDecimal sum) {
        if (sum.compareTo(BigDecimal.ZERO) < 0) {
            return BigDecimal.ONE.negate();
        } else {
            return BigDecimal.ONE;
        }
    }
}
