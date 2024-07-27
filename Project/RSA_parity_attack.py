from Crypto.Util import number
from Crypto.Random import random
from Crypto.Util.number import bytes_to_long, long_to_bytes
import random

def modular_inverse(a, b):
    x, y, x1, y1 = 1, 0, 0, 1
    a1, b1 = a, b

    while b1 != 0:
        q = a1 // b1
        x, x1 = x1, x - q * x1
        y, y1 = y1, y - q * y1
        a1, b1 = b1, a1 - q * b1

    return x

class RSA:
    """Handles RSA key generation, encryption, and decryption."""

    def __init__(self, key_size):
        self.p = number.getPrime(key_size)
        self.q = number.getPrime(key_size)
        self.n = self.p * self.q
        self.phi = (self.p - 1) * (self.q - 1)

        random_extra_bits = random.randint(1, key_size - 1)
        self.e = number.getPrime(key_size + random_extra_bits)
        
        self.d = modular_inverse(self.e, self.phi) % self.phi
        assert self.d > 0 and (self.d * self.e) % self.phi == 1 and number.GCD(self.e, self.phi) == 1

    def encrypt(self, data):
        return pow(bytes_to_long(data), self.e, self.n)

    def decrypt(self, encrypted_data):
        return long_to_bytes(pow(encrypted_data, self.d, self.n)).decode()

class RSAParityOracle(RSA):
    """Extends RSA with a parity check method."""

    def is_odd_parity(self, encrypted_data) -> bool:
        return pow(encrypted_data, self.d, self.n) % 2 == 1

def parity_oracle_attack(ciphertext, oracle: RSAParityOracle):
    low, high = 0, oracle.n - 1
    multiplier = pow(2, oracle.e, oracle.n)
    original_ciphertext = ciphertext

    while low < high:
        mid = (low + high) // 2
        ciphertext = (ciphertext * multiplier) % oracle.n

        if oracle.is_odd_parity(ciphertext):
            low = mid + 1
        else:
            high = mid

    low &= ~0xff
    for i in range(256):
        try:
            if oracle.encrypt(long_to_bytes(low + i)) == original_ciphertext:
                return long_to_bytes(low + i)
        except UnicodeDecodeError:
            pass

def main():
    user_input = input("Enter the message: ")
    print("Original message in bytes:", user_input.encode())

    rsa_oracle = RSAParityOracle(1024)

    encrypted_message = rsa_oracle.encrypt(user_input.encode())
    print("Encrypted message:", encrypted_message)
    decrypted_message = rsa_oracle.decrypt(encrypted_message)
    print("Decrypted message:", decrypted_message)

    assert decrypted_message == user_input

    recovered_plaintext = parity_oracle_attack(encrypted_message, rsa_oracle)
    print("Recovered plaintext:", recovered_plaintext.decode())

    assert recovered_plaintext.decode() == user_input

if __name__ == '__main__':
    main()
