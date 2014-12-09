package com.flowyk.neuron.messenger;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class ActivationInput {

    private final List<BigDecimal> input;

    public ActivationInput(@NotNull List<BigDecimal> input) {
        this.input = Collections.unmodifiableList(input);
    }

    public List<BigDecimal> getInput() {
        return input;
    }
}
