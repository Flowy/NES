package com.flowyk.neuron;

import com.flowyk.neuron.messenger.NeuronOutput;
import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public class PerceptronTrainer extends NeuronTrainer {

    public PerceptronTrainer(@NotNull Neuron neuron) {
        super(neuron);
    }

    @Override
    protected BigDecimal calculateError(NeuronOutput output, BigDecimal desiredOutput) {
        return desiredOutput.subtract(output.getOutput());
    }
}
