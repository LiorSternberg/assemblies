from learning.components.data_set.constructors import create_training_set_from_list, \
    create_test_set_from_list
from learning.components.input import InputStimuli
from learning.components.model import LearningModel
from learning.components.sequence import LearningSequence
from non_lazy_brain import NonLazyBrain


def my_example_simulation():
    """
    We will show step by step how to create the following simulation:

    Input Bit #0          Input Bit #1
          \                    /
        Area A             Area B
            \               /
                 Area C
                   |
                Output

    Sequence will be:
     1. Input Bit #0 -> Area A
        Input Bit #1 -> Area B

     1. 2 Times:
            Input Bit #0 -> Area A
            Input Bit #1 -> Area B
            Area A -> Area A
            Area B -> Area B

     3. [Area A, Area B]  -> Area C

     4. 2 Times:
            [Area A, Area B]  -> Area C
            Area C -> Area C

     5. Area C -> Output
    """
    # Define our args:
    n = 10000
    k = 100
    p = 0.01
    beta = 0.05
    noise_p = 0.001
    output_values = [0, 1, 0, 1]  # defines the binary function we wish to learn

    # Create the brain
    brain = NonLazyBrain(p)
    for area_name in ('A', 'B', 'C'):
        brain.add_area(area_name, 10000, k, beta)

    brain.add_output_area('Output')

    # Define the mapping between bits of the input, and the brain's stimuli
    # so that the model will later know how to translate the input into
    # active stimuli (inactive stimuli don't fire even they are defined in
    # the sequence, they are filtered out using this mapping).
    # (This mapping is also known as InputStimuli)
    input_stimuli = InputStimuli(brain, k, 'A', 'B')

    # Create the sequence of projections to be used in the model
    sequence = LearningSequence(brain, input_stimuli)
    sequence.add_iteration(input_bits=[0, 1])

    sequence.add_iteration(input_bits=[0, 1],
                           areas_to_areas={'A': ['A'], 'B': ['B']},
                           consecutive_runs=2)

    sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})

    sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C'], 'C': ['C']},
                           consecutive_runs=2)

    sequence.add_iteration(areas_to_areas={'C': ['Output']})
    sequence.display_connections_graph()

    # Create the data sets:
    # If we want to define which values are used for training, we need to define
    # a mask.
    # Example: mask = create_explicit_mask_from_list([1, 1, 1, 0])
    # Using this mask will mean only the first 3 inputs are used for training.

    # But let's say we want to use all of the inputs (so we don't provide a
    # a mask, and we will get the default full mask)

    training_set = create_training_set_from_list(
        data_set_return_values=output_values,
        training_set_length=30,
        noise_probability=noise_p
    )

    # Create a test set (with the default full mask) to test all inputs:
    test_set = create_test_set_from_list(output_values)

    # Create the model
    model = LearningModel(brain=brain,
                          sequence=sequence,
                          input_stimuli=input_stimuli)

    model.train_model(training_set=training_set, number_of_sequence_cycles=1)

    test_results = model.test_model(test_set)

    print(
        f"Finished testing the trained model - results:\n"
        f"Accuracy: {test_results.accuracy}\n"
        f"Precision: {test_results.precision}\n"
        f"Recall: {test_results.accuracy}\n"
    )


if __name__ == '__main__':
    my_example_simulation()
