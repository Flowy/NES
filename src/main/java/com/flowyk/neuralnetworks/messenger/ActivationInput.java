package com.flowyk.neuralnetworks.messenger;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class ActivationInput {

    private final List<BigDecimal> input;
    private final BigDecimal bias;

    public ActivationInput(@NotNull List<BigDecimal> input, @NotNull BigDecimal bias) {
        this.input = Collections.unmodifiableList(input);
        this.bias = bias;
    }

    public List<BigDecimal> getInput() {
        return input;
    }
    public BigDecimal getBias() { return bias; }
}
