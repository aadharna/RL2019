import numpy as np
import pickle

ROWS = 3
COLS = 3

BOARD_SIZE = ROWS*COLS

class State:
    def __init__(self):

        self.data = np.zeros((ROWS, COLS))
        self.winner = None
        self.end = None

    def rep(self):
        return(tuple(self.data.reshape(BOARD_SIZE)))

    def is_end(self):
        """

        :return: Boolean: Is the game over.
        """

        if self.end is not None:
            return self.end

        results = []

        #check rows
        for i in range(ROWS):
            results.append(np.sum(self.data[i, :]))

        #check cols
        for i in range(COLS):
            results.append(np.sum(self.data[:, i]))

        results.append(0)
        for i in range(ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for j in range(COLS):
            results[-1] += self.data[j, COLS - 1 - j]

        for each_check in results:
            if each_check == 3:
                self.winner = 1
                self.end = True
                return self.end
            if each_check == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum = np.sum(np.abs(self.data))
        if sum == ROWS * COLS:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def print_state(self):
        for i in range(0, BOARD_SIZE):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_SIZE):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')


def get_all_states_implication(c_state, c_symbol, all_states):
    for i in range(ROWS):
        for j in range(COLS):
            if c_state.data[i][j] == 0:
                n_state = c_state.next_state(i, j, c_symbol)
                n_rep = n_state.rep()
                if n_rep not in all_states.keys():
                    is_end = n_state.is_end()
                    all_states[n_rep] = (n_rep, is_end)

                    if not is_end:
                        #Let the other agent make a move
                        get_all_states_implication(n_state, -c_symbol, all_states)

def get_all_states():

    symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.rep()] = (current_state.rep(), current_state.is_end())
    get_all_states_implication(current_state, symbol, all_states)

    return all_states


t = get_all_states()

#for k in t.keys():
#    print(k, t[k])

print(len(t.keys()))

class Player:
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon

        #relavent states list
        self.states = []

        #Used to store if the move was exploitative of not.
        # True => exploitative
        # False => explorative
        # We do not learn from explorative moves.
        self.exploit = []


    def reset(self):
        self.states = []
        self.exploit = []

    #We assume the move is one we can learn from
    def set_state(self, state):
        self.states.append(state)
        self.exploit.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for each_state in t.keys():
            state, is_end = t[each_state]

            if is_end:

                #if winner is p1
                if state.winner == is_end:
                    self.estimations[state] = 1.0
                #if there is no winner
                elif state.winner == 0:
                    self.estimations[state] = 0
                #if winner is p2
                else:
                    self.estimations[state] = 0
            #if the game has no winner yet
            # set value of the state to 0.5 initially
            else:
                self.estimations[state] = 0.5

    def act(self):
        current_state = self.states[-1]

        next_states = []
        next_position = []

        #Check all states
        for i in range(ROWS):
            for j in range(COLS):
                #Add in all next valid moves
                if current_state.data[i, j] == 0:
                    next_position.append((i, j))
                    next_states.append(current_state.next_state(i, j, self.symbol))

        #Check to see if this next move is an exploratory move

        if np.random.rand() < self.epsilon:
            #Choose a random move
            action = next_position[np.random.randint(len(next_position))]
            #save who made the move
            action.append(self.symbol)
            self.exploit.append(False)
            return action

        value = []
        for rep, pos in zip(next_states, next_position):
            value.append((self.estimations[rep], pos))

        #If the value of each move in the list above (value) is the same,
        # choose a move at random
        np.random.shuffle(value)
        #else, if there is a move with a higher value, choose that
        value.sort(key=lambda x: x[0], reverse=True)

        #extract the top best position
        action = value[0][1]
        action.append(self.symbol)
        return action




    # update value estimation
    def backup(self):
        #for debug
        # print('player trajectory')
        # for state in self.states:
        #     state.print_state()


        move_list = [state.rep() for state in self.states]

        #Since the end of the match holds the winning information
        # start there and percolate up the VALUE of this chain of states.

        #
        # move_list[-1] is the end of the match, therefore it contains
        #  a value estimation (did we win, lose, or tie?)
        #  using that as a starting point, change the value of the state
        #  prior by a small amount, if the move was exploitative).
        #

        for i in reversed(range(len(move_list) - 1)):
            state = move_list[i]

            #I am unsure of if it should be (self.est[move_list[i+1] - self.est[state]
            # or if it should be what I put below.

            td_error = self.exploit[i] * (self.estimations[state] - self.estimations[move_list[i + 1]])
            self.estimations[state] += self.step_size * td_error



    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)






class Judge:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.curr_player = None
        self.p1_sym = 1
        self.p2_sym = -1
        self.p1.set_sym(self.p1_sym)
        self.p2.set_sym(self.p2_sym)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate_play(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self):

        player_alternator = self.alternate_play()
        self.reset()
        current_state = State()

        self.p1.set_state(current_state)
        self.p2.set_state(current_state)

        while True:
            current_player = next(player_alternator)
            [i, j, sym] = current_player.act()
            next_state = current_state.next_state(i, j, sym)
            current_state, is_end = t[next_state.rep()]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)

            if is_end:
                return current_state.winner


