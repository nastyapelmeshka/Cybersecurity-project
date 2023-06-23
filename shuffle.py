import random

# File names and paths to the text files
file1 = 'clean-alexa-32k.txt'
file2 = 'dga-bader-32k.txt'
output_file = 'mixed_files.csv'

# Function to read the text file
def read_text_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    return data

# Read text files
data1 = read_text_file(file1)
data2 = read_text_file(file2)

# Combine the data
combined_data = data1 + data2

# Randomly shuffle the data
random.shuffle(combined_data)

# Write the mixed data to a new text file
with open(output_file, 'w') as file:
    file.writelines(combined_data)

print("The text files have been successfully mixed and saved in '{}'.".format(output_file))
