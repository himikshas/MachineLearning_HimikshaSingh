#!/usr/bin/python

""" Implement ATA  -  A = [1 2 3
                           4 5 6]"""
A = [
    [1, 2, 3],
    [4, 5, 6]
]

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

AT = transpose(A)

def mat(X, Y):
    result = [[0 for _ in range(len(Y[0]))] for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    return result

ATA = mat(AT, A)

for row in ATA:
    print(row)

