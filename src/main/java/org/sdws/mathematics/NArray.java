package org.sdws.mathematics;

import java.util.Random;
import java.util.Objects;
import java.util.Arrays;

public class NArray {
    private float[] vec;
    private float[][] matrix;
    public int elements, dimension;
    public int[] shape;
    private static Random rand = new Random();

    /*
    Class methods
     */

    private NArray(float[] vecInput) {
        this.vec = vecInput;
        this.matrix = null;
        this.dimension = 1;
        this.elements = this.vec.length;
        this.shape = new int[2];
        this.shape[0] = 1;
        this.shape[1] = this.elements;
    }

    private NArray(float[][] matrixInput) {
        this.vec = null;
        this.matrix = matrixInput;
        this.dimension = 2;
        this.shape = new int[2];
        this.shape[0] = this.matrix.length;
        this.shape[1] = this.matrix[0].length;
        this.elements = this.shape[0] * this.shape[1];
    }

    @Override
    public String toString() {
        if (this.matrix == null) {
            return Arrays.toString(this.vec);
        } else {
            return Arrays.deepToString(this.matrix);
        }
    }

    public float get(int index) {
        if (this.vec == null) {
            throw new IllegalStateException("Cannot retrieve elements this way from a matrix.");
        }

        if (Objects.equals(this.elements, 0)) {
            throw new IllegalStateException("Cannot retrieve anything from an empty vector.");
        }

        if (index < 0 || index >= this.vec.length) {
            throw new IndexOutOfBoundsException("Index: " + index + " is out of bounds for this vector NArray.");
        }

        return this.vec[index];
    }

    public void set(int index, float value) {
        if (this.vec == null) {
            throw new IllegalStateException("Cannot retrieve elements this way from a matrix.");
        }

        if (Objects.equals(this.elements, 0)) {
            throw new IllegalStateException("Cannot retrieve anything from an empty vector.");
        }

        if (index < 0 || index >= this.vec.length) {
            throw new IndexOutOfBoundsException("Index: " + index + " is out of bounds for this vector NArray.");
        }

        this.vec[index] = value;
    }

    public float get(int row, int column) {
        if (this.matrix == null) {
            throw new IllegalStateException("Cannot retrieve elements from a vector.");
        }

        if (Objects.equals(this.elements, 0)) {
            throw new IllegalStateException("Cannot retrieve anything from an empty matrix.");
        }

        if (row >= this.matrix.length || row < 0) {
            throw new IndexOutOfBoundsException("Row index: " + row + " is out of bounds for this matrix NArray.");
        }

        if (column >= this.matrix[row].length || column < 0) {
            throw new IndexOutOfBoundsException("Column index: " + column + " is out of bounds for this matrix NArray.");
        }

        return this.matrix[row][column];
    }

    public void set(int row, int column, float value) {
        if (this.matrix == null) {
            throw new IllegalStateException("Cannot retrieve elements from a vector.");
        }

        if (Objects.equals(this.elements, 0)) {
            throw new IllegalStateException("Cannot retrieve anything from an empty matrix.");
        }

        if (row >= this.matrix.length || row < 0) {
            throw new IndexOutOfBoundsException("Row index: " + row + " is out of bounds for this matrix NArray.");
        }

        if (column >= this.matrix[row].length || column < 0) {
            throw new IndexOutOfBoundsException("Column index: " + column + " is out of bounds for this matrix NArray.");
        }

        this.matrix[row][column] = value;
    }

    public float[] getVecArray() {
        if (vec == null) {
            throw new IllegalStateException("Inner vector array is null, did you mean getMatrixArray()?");
        }
        return vec;
    }

    public float[][] getMatrixArray() {
        if (matrix == null) {
            throw new IllegalStateException("Inner matrix array is null, did you mean getVecArray()?");
        }
        return matrix;
    }

    /**
     * Mutates the NArray in place to transpose between matrices and vectors.
     */
    public void transpose() {
        switch (this.dimension) {
            case 1: {   // transpose vector to matrix
                this.vecToMatrix();
                break;
            }
            case 2: {   // transpose matrix to either matrix or vector
                if (this.shape[1] == 1) {
                    this.matrixToVector();
                } else {
                    this.matrixToMatrix();
                }
                break;
            }
            default: {
                break;
            }
        }
    }

    private void vecToMatrix() {
        float[][] matrix = new float[this.elements][1];

        for (int i = 0; i < this.elements; i++) {
            matrix[i][0] = this.vec[i];
        }

        this.matrix = matrix;
        this.vec = null;
        this.dimension = 2;
        this.shape = new int[2];
        this.shape[0] = this.matrix.length;
        this.shape[1] = this.matrix[0].length;
    }

    private void matrixToVector() {
        float[] vec = new float[this.elements];
        for (int i = 0; i < this.elements; i++) {
            vec[i] = this.matrix[i][0];
        }

        this.matrix = null;
        this.vec = vec;
        this.dimension = 1;
        this.shape = new int[2];
        this.shape[0] = 1;
        this.shape[1] = this.elements;
    }

    private void matrixToMatrix() {
        float[][] matrix = new float[this.shape[1]][this.shape[0]];
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                matrix[j][i] = this.matrix[i][j];
            }
        }

        this.matrix = matrix;
        this.dimension = 2;
        this.shape = new int[2];
        this.shape[0] = this.matrix.length;
        this.shape[1] = this.matrix[0].length;
    }

    /*
    Static class functions
     */

    /**
     *
     * @param rows
     * @return
     * @throws IllegalArgumentException only if the length of rows is not equal
     */
    public static NArray FromRows(float[]... rows) throws IllegalArgumentException {
        if (rows.length == 0 || rows[0].length == 0) {
            return new NArray(new float[0][0]);
        }

        if (rows.length == 1) {
            return NArray.FromElements(rows[0]);
        }

        float[][] matrix = new float[rows.length][rows[0].length];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i].length != rows[0].length) {
                throw new IllegalArgumentException(
                        "Input list rows must have the same number of columns for matrix initialization."
                );
            }

            matrix[i] = rows[i].clone();
        }

        return new NArray(matrix);
    }

    public static NArray FromNested(float[][] nestedInput) {
        if (nestedInput.length == 0 || nestedInput[0].length == 0) {
            return null;
        }

        if (nestedInput.length < 2) {
            return NArray.FromElements(nestedInput[0]);
        }

        return new NArray(nestedInput);
    }

    public static NArray FromElements(float... elements) {
        if (elements.length == 0) {
            return new NArray(new float[]{});
        }

        return new NArray(elements);
    }

    public static NArray CreateZeros(int length) {
        float[] input = new float[length];
        Arrays.fill(input, 0f);
        return new NArray(input);
    }

    public static NArray CreateZeros(int rows, int columns) {
        float[][] input = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            Arrays.fill(input[i], 0f);
        }
        return new NArray(input);
    }

    public static NArray CreateRandom(int length) {
        float[] input = new float[length];
        for (int i = 0; i < length; i++) {
            input[i] = (rand.nextFloat() * 2) - 1;
        }

        return new NArray(input);
    }

    public static NArray CreateRandom(int rows, int columns) {
        float[][] input = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                input[i][j] = (rand.nextFloat() * 2) - 1;
            }
        }

        return new NArray(input);
    }

    public static NArray InitializeHe(int rows, int columns) {
        float[][] input = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                input[i][j] = ((float) rand.nextGaussian() * (float)Math.sqrt(2f / (float)columns));
            }
        }

        return new NArray(input);
    }

    public static NArray DeepCopy(NArray array) {
        if (array == null) {
            return null;
        }

        switch (array.dimension) {
            case 1: {
                return new NArray(array.getVecArray().clone());
            }
            case 2: {
                float[][] matrix = new float[array.shape[0]][array.shape[1]];
                float[][] original = array.getMatrixArray();
                for (int i = 0; i < array.shape[0]; i++) {
                    matrix[i] = original[i].clone();
                }
                return new NArray(matrix);
            }
            default: {
                return null;
            }
        }
    }

    public static NArray Add(NArray array1, NArray array2) {
        if (array1 == null || array2 == null) {
            return null;
        }

        if (array1.dimension == array2.dimension) { // either vec + vec or matrix + matrix
            if (array1.dimension == 1) {
                return NArray.operateVecs(array1, array2, true);
            } else {
                return NArray.operateMatrices(array1, array2, true);
            }
        } else { // only matrix + vec under certain circumstances
            return NArray.operMatrixVec(array1, array2, true);
        }
    }

    public static NArray Subtract(NArray array1, NArray array2) {
        if (array1 == null || array2 == null) {
            return null;
        }

        if (array1.dimension == array2.dimension) { // either vec + vec or matrix + matrix
            if (array1.dimension == 1) {
                return NArray.operateVecs(array1, array2, false);
            } else {
                return NArray.operateMatrices(array1, array2, false);
            }
        } else { // only matrix + vec under certain circumstances
            return NArray.operMatrixVec(array1, array2, false);
        }
    }

    public static float Dot(NArray array1, NArray array2) {
        if (array1 == null || array2 == null) {
            throw new IllegalArgumentException("Vectors cannot be null for dot product.");
        }

        if (array1.dimension == array2.dimension) {
            if (array1.dimension == 1) {
                return NArray.dotProduct(array1, array2);
            }
        }

        throw new IllegalArgumentException("Cannot determine dot product with provided NArray arguments.");
    }

    public static NArray Mul(NArray array1, NArray array2) {
        if (array1 == null ||  array2 == null) {
            return null;
        }

        if (array1.dimension == array2.dimension) { // either dot product or matrix product
            if (array1.dimension == 1) {
                throw new IllegalArgumentException(
                        "Incorrect function variation for multiplication (For dot product of vectors refer to NArray.Dot)."
                );
            } else {
                return NArray.matrixProduct(array1, array2);
            }
        } else { // matrix * vec product under certain circumstances
            if (array1.dimension == 1) {
                return NArray.matrixVecProduct(array2, array1);
            } else {
                return NArray.matrixVecProduct(array1, array2);
            }
        }
    }

    /*
    Private static helper functions
     */

    private static NArray operateVecs(NArray vec1, NArray vec2, boolean addition) {
        if (!Arrays.equals(vec1.shape, vec2.shape)) {
            throw new IllegalArgumentException("Cannot add vector arrays that do not have the same number of elements.");
        }

        float[] result = new float[vec1.elements];
        for (int i = 0; i < vec1.elements; i++) {
            if (addition) {
                result[i] = vec1.get(i) + vec2.get(i);
            } else {
                result[i] = vec1.get(i) - vec2.get(i);
            }
        }

        return new NArray(result);
    }

    private static NArray operateMatrices(NArray matrix1, NArray matrix2, boolean addition) {
        if (!Objects.equals(matrix1.shape, matrix2.shape)) {
            throw new IllegalArgumentException("Cannot add matrix arrays that do not have the same shape.");
        }

        float[][] result = new float[matrix1.shape[0]][matrix1.shape[1]];
        for (int i = 0; i < matrix1.shape[0]; i++) {
            for (int j = 0; j < matrix1.shape[1]; j++) {
                result[i][j] = matrix1.get(i, j) + matrix2.get(i, j);
            }
        }

        return new NArray(result);
    }

    private static NArray operMatrixVec(NArray array1, NArray array2, boolean addition) {
        if (!Objects.equals(array1.shape[1], array2.shape[1])) {
            throw new IllegalArgumentException(
                    "Vector length does not match number of matrix columns."
            );
        }

        float[][] result = new float[array1.shape[0]][array1.shape[1]];;

        if (array1.shape[0] > 1) {  // first arg is matrix, second arg is vec
            for (int i = 0; i < array1.shape[0]; i++) {
                for (int j = 0; j < array1.shape[1]; j++) {
                    result[i][j] = array1.get(i, j) + array2.get(i, j);
                }
            }
        } else {    // first arg is vec, second arg is matrix
            for (int i = 0; i < array2.shape[0]; i++) {
                for (int j = 0; j < array2.shape[1]; j++) {
                    result[i][j] = array2.get(i, j) + array1.get(i, j);
                }
            }
        }

        return new NArray(result);
    }

    private static float dotProduct(NArray vec1, NArray vec2) {
        if (!Arrays.equals(vec1.shape, vec2.shape)) {
            throw new IllegalArgumentException("Cannot compute dot product of vectors of different length.");
        }

        if (vec1.shape[0] == 0 || vec1.shape[1] == 0) {
            throw new IllegalArgumentException("Cannot compute dot product of vectors with shape 0.");
        }

        float sum = 0;
        for (int i = 0; i < vec1.elements; i++) {
            sum += vec1.get(i) * vec2.get(i);
        }

        return sum;
    }

    private static NArray matrixProduct(NArray matrix1, NArray matrix2) {
        if (matrix1 == null || matrix2 == null) {
            return null;
        }

        int rows1 = matrix1.getMatrixArray().length;
        int cols1 = matrix1.getMatrixArray()[0].length;

        int rows2 = matrix2.getMatrixArray().length;
        int cols2 = matrix2.getMatrixArray()[0].length;

        boolean valid = cols1 == rows2;

        if (valid) {
            float[][] result = new float[rows1][cols2];
            for (int i = 0; i < rows1; i++) {
                for (int j = 0; j < cols2; j++) {
                    float sum = 0;
                    for (int k = 0; k < cols1; k++) {
                        sum += matrix1.get(i, k) * matrix2.get(k, j);
                    }

                    result[i][j] = sum;
                }
            }

            return new NArray(result);
        }

        throw new IllegalArgumentException(
                "Matrix shapes incompatible: (" + rows1 + " x " + cols1 +
                        ") cannot multiply with (" + rows2 + " x " + cols2 + ")"
        );
    }

    private static NArray matrixVecProduct(NArray matrix, NArray vec) {
        if (matrix.dimension != 2 || vec.dimension != 1) {
            throw new IllegalArgumentException("matrixVecProduct() requires arguments in positional form (matrix × vector).");
        }

        int rows = matrix.shape[0];
        int cols = matrix.shape[1];
        int vectorLength = vec.shape[1];

        if (vectorLength != cols) {
            throw new IllegalArgumentException("Vector length must match matrix columns.");
        }

        float[] result = new float[rows];
        for (int i = 0; i < rows; i++) {
            float sum = 0f;
            for (int j = 0; j < cols; j++) {
                sum += matrix.get(i, j) * vec.get(j);
            }
            result[i] = sum;
        }

        return new NArray(result);
    }
}
