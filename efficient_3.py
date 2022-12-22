import sys
from resource import *
import time
import psutil
# Above imports supports the process_memory function


# Function provided in the project pdf
def process_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss / 1024)
    return memory_consumed


# Hard code alpha and delta values
alpha = {'A': {'A': 0, 'C': 110, 'G': 48, 'T': 94},
         'C': {'A': 110, 'C': 0, 'G': 118, 'T': 48},
         'G': {'A': 48, 'C': 118, 'G': 0, 'T': 110},
         'T': {'A': 94, 'C': 48, 'G': 110, 'T': 0}}
delta = 30


# Generate complete string from base string
def generate_string(base_string, index):
    # Insert copy of previous string after the index
    for i in index:
        base_string = base_string[:i + 1] + base_string + base_string[i + 1:]
    complete_string = base_string
    return complete_string


# Read data from input file
def parse_input(filename):
    with open(filename, 'r') as f:
        base_x = f.readline().rstrip()
        # Get the numbers following base string X
        j = []
        for l in f:
            l = l.rstrip()
            if l.isalpha():
                break
            j.append(int(l))

        base_y = l.rstrip()
        # Get the numbers following base string Y
        k = []
        for l in f:
            l = l.rstrip()
            k.append(int(l))

    # Generate complete string from base string
    complete_x = generate_string(base_x, j)
    complete_y = generate_string(base_y, k)
    return complete_x, complete_y


# Bottom Up
def construct_OPT(X, Y, alpha, delta):
    OPT = [[0 for i in range(len(Y) + 1)] for j in range(len(X) + 1)]

    # initialization
    for i in range(len(X) + 1):
        OPT[i][0] = i * delta
    for i in range(1, len(Y) + 1):
        OPT[0][i] = i * delta

    # Construct DP matrix, bottom up
    for j in range(1, len(Y) + 1):
        for i in range(1, len(X) + 1):
            # 3 cases
            OPT[i][j] = min(OPT[i - 1][j - 1] + alpha[X[i - 1]][Y[j - 1]], OPT[i - 1][j] + delta, OPT[i][j - 1] + delta)
    return OPT
    
    
# Top down
def get_alignment(OPT, X, Y):
    final_x = []
    final_y = []
    i = len(X)
    j = len(Y)

    while i > 0 and j > 0:
        # case 1
        if OPT[i][j] == OPT[i - 1][j - 1] + alpha[X[i - 1]][Y[j - 1]]:
            final_x.append(X[i - 1])
            final_y.append(Y[j - 1])
            i = i - 1
            j = j - 1
        # case 2
        elif OPT[i][j] == OPT[i][j - 1] + delta:
            final_x.append('_')
            final_y.append(Y[j - 1])
            j = j - 1
        # case 3
        else:
            final_x.append(X[i - 1])
            final_y.append('_')
            i = i - 1
    # Fill up rest of the spot if any
    while i > 0:
        final_x.append(X[i - 1])
        final_y.append('_')
        i = i - 1

    while j > 0:
        final_x.append('_')
        final_y.append(Y[j - 1])
        j = j - 1

    # Return the optimal value and X Y alignment
    return [str(OPT[-1][-1]), ''.join(final_x[::-1]), ''.join(final_y[::-1])]



def make_alignment(X, Y, direction):
    m = len(X)
    n = len(Y)

    opt = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize column 0 & row 0 to i*GAP
    opt[0][:] = [i*delta for i in range(n + 1)]
    for i in range(m + 1):
        opt[i][0] = i*delta

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if direction == 0:
                opt[i][j] = min(opt[i-1][j-1] + alpha[X[i-1]][Y[j-1]], opt[i-1][j] + delta, opt[i][j-1] + delta)
            else:
                opt[i][j] = min(alpha[X[m - i]][Y[n - j]] + opt[i - 1][j - 1],
                                opt[i-1][j] + delta, opt[i][j-1] + delta)
    return opt[m]


def dividenconquer(X, Y):
    m = len(X)
    n = len(Y)

    if m < 3 or n < 3:
        Opt, X, Y = get_alignment(construct_OPT(X, Y, alpha, delta), X, Y)
        return Opt, X, Y

    # Half of X and complete Y
    forward = make_alignment( X[:int(m/2)], Y, direction = 0)
    backward = make_alignment( X[int(m/2):], Y, direction = 1)

    # Combine lists
    combine = []
    for i in range(n + 1):
        combine.append(forward[i] + backward[n - i])

    # Find the point that minimize cost
    point = combine.index(min(combine))
    left = dividenconquer( X[:int(m/2)], Y[:point])
    right = dividenconquer( X[int(m/2):], Y[point:])

    result = []
    result.append(int(left[0]) + int(right[0]))
    for i in range(1,len(left)):
        result.append(left[i] + right[i])
    
    return result



if __name__ == '__main__':
        start_time = time.time()
        # Parse input.txt
        X, Y = parse_input(sys.argv[1])
        data = dividenconquer(X, Y)
        end_time = time.time()
        time_taken = (end_time - start_time) * 1000
        memory = process_memory()

        # Write to output file
        output_file = sys.argv[2]
        with open(output_file, "w+") as f:
            for line in data:
                f.write(str(line) + "\n")
            f.write(f"{time_taken}\n")
            f.write(f"{memory}")
            f.close()







