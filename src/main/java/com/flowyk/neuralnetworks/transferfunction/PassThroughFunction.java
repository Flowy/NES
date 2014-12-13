package com.flowyk.neuralnetworks.transferfunction;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;

/**
 * Created by Lukas on 13. 12. 2014.
 */
public class PassThroughFunction implements TransferFunction {
    @Override
    public BigDecimal transfer(@NotNull BigDecimal sum) {
        return sum;
    }
}
