package com.flowyk.neuralnetworks.network;

import com.flowyk.neuralnetworks.messenger.ActivationInput;
import com.flowyk.neuralnetworks.messenger.NeuronOutput;
import com.flowyk.neuralnetworks.neuron.Neuron;

import java.math.BigDecimal;
import java.util.List;

/**
 * Created by Lukas on 13. 12. 2014.
 */
class InputNeuron implements Neuron {

    @Override
    public void train(ActivationInput input, BigDecimal error) {
        throw new UnsupportedOperationException("Training is not allowed on InputNeuron");
    }

    @Override
    public NeuronOutput activate(ActivationInput input) {
        return null;
    }

    @Override
    public List<BigDecimal> getWeights() {
        return null;
    }
}
