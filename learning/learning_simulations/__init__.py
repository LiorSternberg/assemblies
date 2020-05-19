from enum import Enum, auto


class Strategy(Enum):
    """
    Defining the strategy of simulation builder:
        1) Simple: all stimuli directly fire at a single area (which fires at the output area)
        2) Layered: every two 'sibling' stimuli fires at a unique area ('first layer'), which, together with
            its sibling area, fires at a unique area ('second layer'), until it finally converges into a single
            area ('last layer') - which fires at the output area. An illustration (for domain size of 2):

                    Stimulus A   Stimulus B         Stimulus C   Stimulus D ...
                       \            /                     \       /
                           Area A                          Area B
                               \                             /
                                         Area C
                                           |
                                        Output
    """
    Simple = auto()
    Layered = auto()
