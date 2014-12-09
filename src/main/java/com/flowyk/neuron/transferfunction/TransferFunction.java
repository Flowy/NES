package com.flowyk.neuron.transferfunction;

import java.math.BigDecimal;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public interface TransferFunction {
    public BigDecimal transfer(BigDecimal sum);
}
