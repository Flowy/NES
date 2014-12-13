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
        private List<Neuron> inputNeurons;
        private Neuron outputNeuron;

        public NeuronPipe(@NotNull List<Neuron> inputNeurons, @NotNull Neuron outputNeuron) {
            this.inputNeurons = inputNeurons;
            this.outputNeuron = outputNeuron;
        }

        public boolean isInInput(Neuron neuron) {
            boolean result = false;
            for (Neuron n: inputNeurons) {
                if (n == neuron) {
                    result = true;
                    break;
                }
            }
            return result;
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
