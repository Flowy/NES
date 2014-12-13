package com.flowyk.neuralnetworks.messenger;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class TrainingInput extends ActivationInput {

    private final BigDecimal desiredOutput;

    public TrainingInput(@NotNull List<BigDecimal> input, @NotNull BigDecimal bias, @NotNull BigDecimal desiredOutput) {
        super(input, bias);
        this.desiredOutput = desiredOutput;
    }

    public BigDecimal getDesiredOutput() {
        return desiredOutput;
    }
}