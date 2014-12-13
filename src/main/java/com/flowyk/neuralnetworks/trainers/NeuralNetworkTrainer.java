package com.flowyk.neuralnetworks.trainers;

import com.flowyk.neuralnetworks.messenger.TrainingInput;

import java.util.List;

/**
 * Created by Lukas on 13. 12. 2014.
 */
public interface NeuralNetworkTrainer {

    public void trainAll(List<TrainingInput> inputs);

}
