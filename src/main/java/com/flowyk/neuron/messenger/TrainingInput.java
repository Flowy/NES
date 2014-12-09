package com.flowyk.neuron.messenger;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class TrainingInput extends ActivationInput {

    private final BigDecimal output;

    public TrainingInput(@NotNull List<BigDecimal> input, @NotNull BigDecimal output) {
        super(input);
        this.output = output;
    }

    public BigDecimal getDesiredOutput() {
        return output;
    }
}