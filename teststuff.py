import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols

import TSST as TT



data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]}


def calculate_mean_and_std(df):
    # Calculate mean and standard deviation for each column
    mean_values = df.mean()
    std_values = df.std()

    # Create a new DataFrame with mean and std for each column
    result_df = pd.DataFrame({
        'Mean': mean_values,
        'Std': std_values
    })

    return result_df.transpose()

# Example usage:
df = pd.DataFrame(data)

result = calculate_mean_and_std(df)
print(result)


exit()

df = pd.DataFrame(data)


M=2

#wanted output  A ~ B + C + A**2 + B**2 + A*B

from itertools import permutations, product


# Example usage:
words = ['wordA', 'wordB', 'wordC']
#print([permutation for permutation in permutations(words,r=2)])
#print([permutation for permutation in product(words)])


words_copy=words.copy()

for word in words:
    print(word)
    words_copy2=words_copy.copy()

    for word2 in words_copy:
        print(word+"*"+word2)

        for word3 in words_copy2:
            print(word+"*"+word2+"*"+word3)

        words_copy2.remove(word2)

    words_copy.remove(word)

print("\n")

def word_iterator(string,words,M,comb):
    print(string)
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            print(string+"*"+word)
            comb.append(string+"*"+word)
        else: 
            word_iterator(string+"*"+word,words_copy,M-1,comb)
        words_copy.remove(word)

def word_multiplicator(words,M=2):
    comb=[]
    words_copy=words.copy()
    for word in words:
        word_iterator(word,words_copy,M-1,comb)
        words_copy.remove(word)
    return comb

comb = word_multiplicator(words,M=3)

print(comb)
#result = generate_word_combinations(words)
#print(result)
