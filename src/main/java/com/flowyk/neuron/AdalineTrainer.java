package com.flowyk.neuron;

import com.flowyk.neuron.messenger.NeuronOutput;
import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public class AdalineTrainer extends NeuronTrainer {

    public AdalineTrainer(@NotNull Neuron neuron) {
        super(neuron);
    }

    @Override
    protected BigDecimal calculateError(NeuronOutput output, BigDecimal desiredOutput) {
        BigDecimal outputSum = BigDecimal.ZERO;
        for (BigDecimal sensorOutput : output.getSensorOutputs()) {
            outputSum = outputSum.add(sensorOutput);
        }
        return desiredOutput.subtract(outputSum);
    }
}
