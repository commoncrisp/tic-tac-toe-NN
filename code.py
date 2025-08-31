import numpy as np
import itertools
import math
import os  # for checking if weight files exist

# Board mapping
mapping = {'X': -1, 'O': 1, ' ': 0}

def print_board():
    print(f"{board[0]}|{board[1]}|{board[2]}")
    print("-+-+-")
    print(f"{board[3]}|{board[4]}|{board[5]}")
    print("-+-+-")
    print(f"{board[6]}|{board[7]}|{board[8]}")

# Check winner
def check_winner(b):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for i,j,k in wins:
        if b[i] == b[j] == b[k] != ' ':
            return b[i]
    if ' ' not in b:
        return 'Draw'
    return None

# Minimax
def minimax(b, is_maximizing):
    winner = check_winner(b)
    if winner == 'O': return 1
    if winner == 'X': return -1
    if winner == 'Draw': return 0
    
    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'O'
                score = minimax(b, False)
                b[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if b[i] == ' ':
                b[i] = 'X'
                score = minimax(b, True)
                b[i] = ' '
                best_score = min(score, best_score)
        return best_score

# AI move vector
def ai_move_vector(board):
    scores = [0] * 9
    best_score = -math.inf
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, False)
            board[i] = ' '
            scores[i] = score
            if score > best_score:
                best_score = score
    return np.array([1 if s == best_score else 0 for s in scores])

# Generate training set
def generate_training_data():
    boards = []
    targets = []
    for x_count in range(6):
        for positions in itertools.combinations(range(9), x_count):
            for o_count in range(x_count + 1):
                for o_positions in itertools.combinations(positions, o_count):
                    board = [' '] * 9
                    for pos in positions:
                        board[pos] = 'X'
                    for pos in o_positions:
                        board[pos] = 'O'
                    if check_winner(board) is None:
                        boards.append([mapping[s] for s in board])
                        targets.append(ai_move_vector(board))
    return np.array(boards), np.array(targets)

# Create training data
X, Y = generate_training_data()
print("Training samples:", X.shape[0])

# Neural network
neurons = [9, 128, 128, 9]
lr0, min_lr = 0.07, 0.007
lr = lr0
epochs = 10000

# Check if saved weights exist
if os.path.exists("model.npz"):
    print("Loading saved weights and biases...")
    data = np.load("model.npz", allow_pickle=True)
    weights = data["weights"]
    biases = data["biases"]

else:
    print("Initializing new weights and biases...")
    weights = [np.random.randn(neurons[i-1], neurons[i]) * np.sqrt(2/neurons[i-1]) for i in range(1, len(neurons))]
    biases = [np.random.randn(neurons[i]) for i in range(1, len(neurons))]

def leaky_relu(x, alpha=0.01):
    return np.tanh(x)
def leaky_relu_derivative(x, alpha=0.01):
    return 1 - np.tanh(x)**2

# Training
for epoch in range(epochs):
    a = [X]
    z = []
    for ii in range(len(neurons)-1):
        Z = a[-1] @ weights[ii] + biases[ii]
        A = leaky_relu(Z)
        z.append(Z)
        a.append(A)
    
    y_pred = a[-1]
    error = y_pred - Y
    loss = np.mean(error**2)

    dZ = []
    dW = []
    dB = []

    for ii in range(len(neurons)-1):
        layer = -(ii+1)
        if layer == -1:
            DZ = (2/X.shape[0]) * error
        else:
            DA = dZ[-1] @ weights[layer+1].T
            DZ = DA * leaky_relu_derivative(z[layer])
        if layer == -(len(neurons)-1):
            DW = X.T @ DZ
        else:
            DW = a[layer-1].T @ DZ
        DB = np.sum(DZ, axis=0)
        dZ.append(DZ)
        dW.append(DW)
        dB.append(DB)
    
    for ii in range(len(weights)):
        weights[ii] -= lr * dW[-(ii+1)]
        biases[ii] -= lr * dB[-(ii+1)]
    
    lr = min_lr + 0.5*(lr0-min_lr)*(1 + np.cos(np.pi*epoch/epochs))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save weights and biases after training
np.savez("model.npz", weights=np.array(weights, dtype=object), biases=np.array(biases, dtype=object))

print("Weights and biases saved!")

# Neural network move function
def neural_network_move(board):
    x = np.array([mapping[s] for s in board]).reshape(1, 9)
    a = []
    z = []
    for ii in range(len(neurons) - 1):
        if not a:
            Z = x @ weights[ii] + biases[ii]
        else:
            Z = a[-1] @ weights[ii] + biases[ii]
        A = leaky_relu(Z)
        z.append(Z)
        a.append(A)

    y_pred = a[-1].flatten()

    # Mask invalid moves
    y_pred_masked = np.copy(y_pred)
    for i, val in enumerate(board):
        if val != ' ':
            y_pred_masked[i] = -np.inf

    move = np.argmax(y_pred_masked)
    print(y_pred)
    return move

keep_going = True
while keep_going:
    board = [' '] * 9
    while True:
        valid = False
        while not valid:
            try:
                pos = int(input("Enter your move (0-8): "))
                if board[pos] == ' ':
                    board[pos] = 'X'
                    valid = True
                else:
                    print("Position already taken.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number 0-8.")

        print_board()
        winner = check_winner(board)
        if winner:
            break

        move = neural_network_move(board)
        board[move] = 'O'
        print("AI played:")
        print_board()

        winner = check_winner(board)
        if winner:
            break

    if winner == 'Draw':
        print("It's a draw!")
    else:
        print(f"{winner} wins!")
