import numpy as np
from math import sqrt, ceil

# Input key and message
key = input("Please Enter the Key: ").strip().upper()
message = input("Please Enter the Message: ").strip().upper()

# Calculate key matrix size
key_size = int(sqrt(len(key)))
message_len = len(message)

# Ensure the key is a square
if key_size * key_size != len(key):
    print("Key is not square")
    exit()

# Pad the message if necessary
if len(message) % key_size != 0:
    message = message + "X" * (key_size - len(message) % key_size)

# Create key matrix
key_matrix = np.array([ord(char) - 65 for char in key]).reshape(key_size, key_size)

# Create message matrix
message_matrix = np.array([ord(char) - 65 for char in message]).reshape(len(message) // key_size, key_size).T

# Multiply and mod the matrices
result_matrix = np.dot(key_matrix, message_matrix) % 26

# Convert result to characters and flatten
result = "".join([chr(char + 65) for char in result_matrix.flatten(order='F')])

# Print the encrypted message
print(result)
