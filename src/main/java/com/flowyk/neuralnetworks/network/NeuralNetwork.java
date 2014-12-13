package com.flowyk.neuralnetworks.network;

import com.flowyk.neuralnetworks.neuron.Neuron;
import com.sun.istack.internal.NotNull;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Lukas on 13. 12. 2014.
 */
public class NeuralNetwork {
    private static class NeuronPipe {
        private Neuron inputNeuron;
        private Neuron outputNeuron;

        public NeuronPipe(@NotNull Neuron inputNeuron, @NotNull Neuron outputNeuron) {
            this.inputNeuron = inputNeuron;
            this.outputNeuron = outputNeuron;
        }

        public boolean isInput(Neuron neuron) {
            return this.inputNeuron == neuron;
        }

        public boolean isOutput(Neuron neuron) {
            return this.outputNeuron == neuron;
        }
    }

    private List<Neuron> neurons = new ArrayList<>();

    public NeuralNetwork(@NotNull Neuron neuron) {
        neurons.add(neuron);
    }


}
