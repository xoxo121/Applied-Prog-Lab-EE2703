from typing import List, Tuple

def matrix_multiply(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    # Function to get the shape of the matrix
    def get_shape(matrix: List[List[float]]) -> Tuple[int, int]: 
        # Raise an error if the matrix is empty
        if len(matrix) == 0 or len(matrix[0]) == 0:
            raise ValueError("Matrix cannot be empty")
        else:
            return (len(matrix), len(matrix[0]))

    shape1 = get_shape(matrix1)
    shape2 = get_shape(matrix2)

    # Raise an error if the matrix dimensions are not compatible
    if shape1[1] != shape2[0]:  
        raise ValueError("Matrix dimensions not compatible for multiplication")
    
    # Return the product of two 1x1 matrices
    if shape1 == (1, 1) and shape2 == (1, 1):
        return [[matrix1[0][0] * matrix2[0][0]]]
    
    # Initialize the result matrix with zeroes
    result = [[0 for i in range(shape2[1])] for j in range(shape1[0])]
    
    # Implement matrix multiplication
    for i in range(shape1[0]):
        for j in range(shape2[1]):
            for k in range(shape2[0]):
                # Raise an error if the matrix elements are not numeric
                if not isinstance(matrix1[i][k], (int, float)) or not isinstance(matrix2[k][j], (int, float)):
                    raise TypeError("Matrix elements must be numeric")
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result
