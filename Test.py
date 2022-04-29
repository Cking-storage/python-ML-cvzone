import numpy as np
'''
A = np.arange(25).reshape(5, 5)
diag = np.einsum('ii->i', A)
trace = np.einsum('ii->', A)
print(A)
print(diag)
print(trace)
'''
'''
A = np.arange(25).reshape(5, 5)
diag = np.einsum('ij', A)
C = np.einsum('ij->i', A)
print(diag)
print(C)
'''
'''
v = np.random.randint(low = 1, high = 10, size = (5, 3))
print(v)
print('\n')

L1_norm = np.linalg.norm(v, axis = 1, ord = 1)
print(L1_norm)
print("sharp : ", L1_norm.shape)    # 축의 순서(첫번째 축, 두번째 축, 세번째 축,...)
print('\n')

L2_norm = np.linalg.norm(v, axis = 1)
print(L2_norm)
print("sharp : ", L2_norm.shape)
print('\n')
'''
'''
v1 = np.random.randint(low = 1, high = 10, size = (5, 3))
v2 = np.random.randint(low= 1, high = 10, size = (5,3))
print(v1)
print(v2)
v = v1/v2
print(v)
'''
'''
A = np.array([0, 1, 2])
B = np.array([[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11]])
C = np.array([3, 4, 5])
print((A[:, np.newaxis] * B).sum(axis=1))
print(np.einsum('i,ij', A, B))
print(np.einsum('ij,i', B, A))
print(np.einsum('i,ij->i', A, B))
print(np.einsum('ij,i->i', B, A))
print(np.einsum('i,i', A, C))
'''
'''
A = np.arange(9).reshape(3,3)
B = np.arange(9, 18).reshape(3,3)
dot = np.einsum('ij, jk->ik', A, B)     # 내적(행렬곱)
dot1 = np.einsum('ij, ij->i', A, B)     # 같은 행의 열끼리 곱한 후 더하여 i 성분 벡터를 만든다.
dot2 = np.einsum('ij, ij->i', B, A)
print(A)
print(B)
print(dot)
print(dot1)
print(dot2)
'''
'''
A = np.array([0, 1, 2])
B = np.array([3, 4, 5])
C = np.array([[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]])
#print(np.einsum('i,i', A, B))
#print(np.einsum('i,ij', A, C))

a = np.arange(9).reshape(3,3)
b = np.arange(3)
print(a)
print(b)
print(np.einsum('ij,i', a, b))      # ?
print(np.einsum('ij,j', a, b))      # 행렬,배열 곱

print(np.einsum('ij,i->i', a, b))   # a 열의 합 x b 열 [3 12 21] x [0 1 2]
print(np.einsum('ij,i->', a, b))    # 출력레이블이 없으면 단일 숫자
'''

A = np.arange(6).reshape(3,2)
B = np.arange(6,12).reshape(3,2)
print(A)
print(B)
print(np.einsum('ij,ij->i', A, B))
