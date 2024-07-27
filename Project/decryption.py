import numpy as np
from math import gcd

# Function for performing Gaussian elimination to solve linear systems of equations
def gaussian_elimination(matrix_a, matrix_b):
    size = len(matrix_a)  # Number of equations/variables
    augmented_matrix = np.hstack((matrix_a, matrix_b)).astype(float)  # Create the augmented matrix
    
    # Forward Elimination process
    for i in range(size):
        if augmented_matrix[i][i] == 0.0:  # Handle zero pivot element
            for j in range(i + 1, size):
                if augmented_matrix[j][i] != 0.0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break

        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]  # Normalize pivot row

        for j in range(i + 1, size):  # Eliminate below pivot
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]

    # Backward Elimination process
    for i in range(size - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j][i]
            
    # Extract solution matrix
    solution_matrix = augmented_matrix[:, size:]
    return solution_matrix

# Function to convert text into a numerical matrix with padding if necessary
def text_to_matrix(text):
    text_length = len(text)
    if text_length % 3 != 0:
        text += "X" * (3 - text_length % 3)  # Padding with 'X' to make the length a multiple of 3
    text_matrix = np.array([ord(char) - 65 for char in text]).reshape(-1, 3)  # Convert text to numbers
    return text_matrix

# Function to find the key matrix using Gaussian elimination
def get_key_matrix(plain_mat, cipher_mat):
    num_columns = len(cipher_mat)
    found = False
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            for k in range(j + 1, num_columns):
                sub_matrix = np.column_stack((plain_mat[i], plain_mat[j], plain_mat[k])).astype(float)
                det = round(np.linalg.det(sub_matrix))
                if gcd(det, 26) == 1:  # Determinant must be coprime with 26
                    cofactor_matrix = np.matrix([[(sub_matrix[(row+1)%3][(col+1)%3] * sub_matrix[(row+2)%3][(col+2)%3] - sub_matrix[(row+1)%3][(col+2)%3] * sub_matrix[(row+2)%3][(col+1)%3]) for row in range(3)] for col in range(3)]).astype(int) % 26
                    temp_key = np.column_stack((cipher_mat[i], cipher_mat[j], cipher_mat[k])).astype(int) % 26
                    temp_key = np.matmul(temp_key, cofactor_matrix)
                    temp_key *= pow(det, -1, 26)
                    temp_key %= 26
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        return temp_key
    else:
        return None

# Function to find the key matrix by brute force
def brute_force_search(plain_mat, cipher_mat, row_idx):
    possible_keys = []
    target_vector = cipher_mat.T[row_idx]
    for col in range(len(cipher_mat)):
        for idx in range(3):
            if plain_mat[col][idx] % 2 == 0 or plain_mat[col][idx] % 13 == 0:
                continue
            if idx == 0:
                row1, row2 = 1, 2
            elif idx == 1:
                row1, row2 = 0, 2
            else:
                row1, row2 = 0, 1
            for i in range(26):
                for j in range(26):
                    k = ((cipher_mat[col][row_idx] - i * plain_mat[col][row1] - j * plain_mat[col][row2]) * pow(int(plain_mat[col][idx]), -1, 26)) % 26
                    if idx == 0:
                        key_vec = np.array([k, i, j])
                    elif idx == 1:
                        key_vec = np.array([i, k, j])
                    else:
                        key_vec = np.array([i, j, k])
                    if np.array_equal(np.matmul(key_vec, plain_mat.T) % 26, target_vector):
                        possible_keys.append(key_vec)
            return possible_keys
    for i in range(26):
        for j in range(26):
            for k in range(26):
                key_vec = np.array([i, j, k])
                if np.array_equal(np.matmul(key_vec, plain_mat.T) % 26, target_vector):
                    possible_keys.append(key_vec)
    return possible_keys

# Main function to execute the process
def main():
    # Take input from the user
    plaintext = input("Enter the plaintext: ").upper()
    ciphertext = input("Enter the ciphertext: ").upper()
    
    # Convert the plaintext and ciphertext into numerical matrices
    plain_matrix = text_to_matrix(plaintext)
    cipher_matrix = text_to_matrix(ciphertext)
    
    # Try to find the key matrix using Gaussian elimination
    key_matrix = get_key_matrix(plain_matrix, cipher_matrix)
    if key_matrix is not None:
        print("Key Matrix:\n", key_matrix)
        key_matrix = np.array(key_matrix)  # Convert to ndarray for rounding
        key_string = "".join([chr(int(round(char)) + 65) for char in key_matrix.flatten()])
        print("Key String:", key_string)
        return
    
    # If key matrix is not found, perform brute force search
    potential_key_matrices = []
    for row in range(3):
        potential_key_matrices.append(brute_force_search(plain_matrix, cipher_matrix, row))
    
    # Print all potential key matrices found
    for key_1 in potential_key_matrices[0]:
        for key_2 in potential_key_matrices[1]:
            for key_3 in potential_key_matrices[2]:
                combined_key = np.column_stack((key_1, key_2, key_3))
                print("Potential Key Matrix:\n", combined_key)

if __name__ == "__main__":
    main()
