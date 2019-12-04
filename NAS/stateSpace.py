from collections import OrderedDict
import numpy as np
import pprint

class StateSpace:
    '''
        State Space manager
        Provides utilit functions for holding "states" / "actions" that the controller
        must use to train and predict.
        Also provides a more convenient way to define the search space

        Each state / action is a dictionnary storing metadata
    '''

    def __init__(self):
        self.states = OrderedDict()
        self.state_count_ = 0
        self.value_count_ = 0

    def add_state(self, name, values):
        '''
            Adds a "state" to the state manager, along with some metadata for efficient
            packing and unpacking of information required by the RNN Controller.
            Stores metadata such as:
            -   Global ID
            -   Name
            -   Valid Values
            -   Number of valid values possible
            -   Map from value ID to state value
            -   Map from state value to value ID

        :param name: name of the state / action
        :param values: valid values that this state can take
        :return: Global ID of the state. Can be used to refer to this state later.
        '''

        index_map = {} # map from value ID to state value
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {} # map from state value to value ID
        for i, val in enumerate(values):
            value_map[val] = i

        embedding_id = []
        for i in range(len(values)):
            embedding_id.append(self.value_count_)
            self.value_count_ += 1

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
            'embedding_id': embedding_id
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def onehot_encode(self, id, value):
        '''
            One-hot encode the specific state value

        :param id: global id of the state
        :param value: state value
        :return: one-hot encoding of the state value
        '''

        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]

        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = 1
        return one_hot

    def get_state_sizes(self, num_layers):
        state_sizes = []

        for id in range(self.size * num_layers):
            state = self[id]
            state_sizes.append(state['size'])

        return state_sizes

    def get_state_value(self, id, index):
        '''
            Retrieves the state value from the state value ID

        :param id: global id of the state
        :param index: index of the state value (usually from argmax)
        :return: The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return value

    def get_embedding_id(self, id, index):
        state = self[id]
        embedding_id = state['embedding_id'][index]
        return embedding_id

    def get_random_state_space(self, num_layers):
        '''
            Constructs a random initial state space for feeding as an initial value to the Controller RNN
            (for each state, randomly selects a state value and one-hot encodes it)

        :param num_layers: number of layers to duplicate the search space
        :return: A list of one hot encoded states
        '''

        states = []

        for id in range(self.size * num_layers):
            state = self[id]
            size = state['size']

            sample = np.random.choice(size, size=1)[0]
            # sample = state['index_map_'][sample[0]]
            # state = self.onehot_encode(id, sample)
            states.append(sample)
        return states

    def parse_state_space_list(self, state_list):
        '''
            Parses a list of one hot encoded states to retrieve a list of state values
        :param state_list: list of one hot encoded states
        :return: list of state values
        '''

        state_values = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            value = self.get_state_value(id, state_val_idx)
            state_values.append(value)

        return state_values

    def get_embedding_ids(self, state_list):
        embedding_ids = []

        for id, ind in enumerate(state_list):
            # state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            embedding_id = self.get_embedding_id(id, ind)
            embedding_ids.append(embedding_id)

        return embedding_ids



    def print_state_space(self):
        ''' Pretty print the state space '''
        print('*' * 40, 'STATE SPACE', '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        ''' Print the action space properly '''
        print('Actions :')

        for id, action in enumerate(actions):
            if id % self.size == 0:
                print("*" * 20, "Layer %d" % (((id + 1) // self.size) + 1), "*" * 20)

            state = self[id]
            name = state['name']
            vals = [(n, p) for n, p in zip(state['values'], action)]
            print("%s : " % name, vals)
        print()

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_


