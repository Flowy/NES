package com.flowyk.neuron.messenger;

import com.sun.istack.internal.NotNull;

import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class TrainingInput extends ActivationInput {

    private final Number output;

    public TrainingInput(@NotNull List<Number> input, @NotNull Number output) {
        super(input);
        this.output = output;
    }

    public Number getDesiredOutput() {
        return output;
    }
}