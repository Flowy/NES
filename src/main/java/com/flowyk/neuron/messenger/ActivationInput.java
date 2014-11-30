package com.flowyk.neuron.messenger;

import com.sun.istack.internal.NotNull;

import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 30. 11. 2014.
 */
public class ActivationInput {

    private final List<Number> input;

    public ActivationInput(@NotNull List<Number> input) {
        this.input = Collections.unmodifiableList(input);
    }

    public List<Number> getInput() {
        return input;
    }
}
