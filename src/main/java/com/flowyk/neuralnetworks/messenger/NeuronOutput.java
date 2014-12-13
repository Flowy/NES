package com.flowyk.neuralnetworks.messenger;

import com.sun.istack.internal.NotNull;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.List;

/**
 * Created by Lukas on 11. 12. 2014.
 */
public class NeuronOutput {
    private final List<BigDecimal> sensorOutputs;
    private final BigDecimal output;

    public NeuronOutput(@NotNull List<BigDecimal> sensorOutputs, @NotNull BigDecimal output) {
        this.sensorOutputs = Collections.unmodifiableList(sensorOutputs);
        this.output = output;
    }

    public List<BigDecimal> getSensorOutputs() {
        return sensorOutputs;
    }

    public BigDecimal getOutput() {
        return output;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof NeuronOutput)) return false;

        NeuronOutput that = (NeuronOutput) o;

        if (!sensorOutputs.equals(that.sensorOutputs)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return sensorOutputs.hashCode();
    }

    @Override
    public String toString() {
        return "NeuronOutput{" +
                "sensorOutputs=" + sensorOutputs +
                ", output=" + output +
                '}';
    }
}
