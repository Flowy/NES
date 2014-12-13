package com.flowyk.neuralnetworks.transferfunction;

import java.math.BigDecimal;

/**
 * Created by Lukas on 6. 12. 2014.
 */
public class BipolarnaFunkcia implements TransferFunction {

    private BigDecimal threshold;

    public BipolarnaFunkcia(BigDecimal threshold) {
        this.threshold = threshold;
    }

    @Override
    public BigDecimal transfer(BigDecimal sum) {
        int compareResult = sum.compareTo(threshold);
        if (compareResult > 0) {
            return BigDecimal.ONE;
        } else if (compareResult < 0) {
            return BigDecimal.ONE.negate();
        } else {
            return BigDecimal.ZERO;
        }
    }
}
