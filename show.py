from util import grader, theta_image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str, help='Path to the file')
args = parser.parse_args()

file_path = args.file_path

B = np.load(file_path)['B']
# print(B)
grader(B)
theta_image(B, path = "tmp.png")