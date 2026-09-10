package org.sdws.mathematics;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class NNMatrix<E> {
    private ArrayList<ArrayList<E>> matrix;
    public ArrayList<Integer> shape;

    private static final Random rng = new Random();

    private NNMatrix(ArrayList<ArrayList<E>> matrix) {
        this.matrix = matrix;
        this.shape = new ArrayList<>(2);
        this.shape.add(this.matrix.get(0).isEmpty() ? 0 : this.matrix.size());
        this.shape.add(this.matrix.get(0).size());
    }

    /*
    Class methods
     */

    @Override
    public String toString() {
        return this.matrix.toString();
    }

    public void set(int row, int column, E value) {
        this.matrix.get(row).set(column, value);
    }

    public ArrayList<E> get(int row) {
        if (row < 0 || row >= this.matrix.size()) {
            throw new IndexOutOfBoundsException(
                    "Row index: " + row + " is out of bounds of this matrix."
            );
        }
        
        return this.matrix.get(row);
    }

    public E get(int row, int column) {
        if (row < 0 || row >= this.matrix.size()) {
            throw new IndexOutOfBoundsException(
                    "Row index: " + row + " is out of bounds of this matrix."
            );
        }
        
        if (column < 0 || column >= this.matrix.get(row).size()) {
            throw new IndexOutOfBoundsException(
                    "Column index: " + column + " is out of bounds of this matrix."
            );
        }
        
        return this.matrix.get(row).get(column);
    }

    public void transpose() {
        ArrayList<ArrayList<E>> result = new ArrayList<>(this.shape.get(1));

        int newRows = this.shape.get(1);
        int newColumns = this.shape.get(0);

        for (int i = 0; i < newRows; i++) {
            ArrayList<E> row = new ArrayList<>();
            for (int j = 0; j < newColumns; j++) {
                row.add((E) (Float) 0f);
            }

            result.add(row);
        }

        for (int i = 0; i < this.shape.get(0); i++) {
            for (int j = 0; j < this.shape.get(1); j++) {
                result.get(j).set(i, this.get(i, j));
            }
        }

        this.matrix = result;
        this.shape.set(0, newRows);
        this.shape.set(1, newColumns);
    }

    /*
    Static class functions
     */

    /**
     * Create a new matrix based on a list of provided arguments of type {@link List}.
     * @param input any amount of {@link List} objects containing
     * elements of the provided type parameter, in the form of a {@code varargs} argument.<br></br>
     * Essentially, the amount of arguments passed corresponds to the amount of rows
     * the final {@link NNMatrix} object will have.
     * @return a new {@link NNMatrix} object.
     * @param <E> the type of the elements of this matrix.
     */
    @SafeVarargs
    public static <E> NNMatrix<E> FromRows(List<E>... input) {
        ArrayList<ArrayList<E>> matrix = new ArrayList<>();

        if (input.length == 0 || input[0].isEmpty()) {
            matrix.add(new ArrayList<>());
            return new NNMatrix<>(matrix);
        }

        for (List<E> row : input) {
            if (!Objects.equals(row.size(), input[0].size())) {
                throw new IllegalArgumentException(
                        "Input list rows must have the same number of columns for matrix."
                );
            }

            matrix.add(new ArrayList<>(row));
        }
        return new NNMatrix<>(matrix);
    }

    public static NNMatrix<Float> CreateZeros(int rows, int columns) {
        return NNMatrix.randomized(rows, columns, 0);
    }

    public static NNMatrix<Float> CreateRandom(int rows, int columns) {
        return NNMatrix.randomized(rows, columns, 1);
    }

    public static NNMatrix<Float> FromHEInitialization(int rows, int columns) {
        return NNMatrix.randomized(rows, columns, 2);
    }

    public static <E> NNMatrix<E> DeepCopy(NNMatrix<E> matrix) {
        if (matrix == null) {
            return null;
        }
        
        ArrayList<ArrayList<E>> result = new ArrayList<>(matrix.shape.get(0));

        for (int i = 0; i < matrix.shape.get(0); i++) {
            ArrayList<E> origRow = matrix.get(i);
            ArrayList<E> copyRow = new ArrayList<>(matrix.shape.get(2));
            
            for (E element : origRow) {
                copyRow.add(element);
            }

            result.add(copyRow);
        }

        return new NNMatrix<>(result);
    }

    /**
     * Only allows transposition of vectors of size 1 x {@code N} into a {@code N} x 1 matrix
     * (a matrix with a single column and {@code N} rows).
     * @param vector a {@link NNVector} object
     * @return a {@link NNMatrix} object
     * @param <E> the type of the elements of the matrix, which must be the same as the vector argument.
     */
    public static <E> NNMatrix<E> TransposeVecToMatrix(NNVector<E> vector) {
        if (vector == null || vector.size() < 1) {
            return null;
        }

        ArrayList<ArrayList<E>> result = new ArrayList<>(vector.size());
        for (int i = 0; i < vector.size(); i++) {
            ArrayList<E> row = new ArrayList<>(1);
            row.add(vector.get(i));
            result.add(row);
        }

        return new NNMatrix<>(result);
    }

    public static <E extends Number> NNMatrix<E> Add(NNMatrix<E> matrix1, NNMatrix<E> matrix2) {
        if (matrix1 == null) {
            return matrix2;
        } else if (matrix2 == null) {
            return matrix1;
        }

        if (!Objects.equals(matrix1.shape, matrix2.shape)) {
            throw new IllegalArgumentException(
                    "Matrices provided do not have the same number of rows and columns."
            );
        }

        ArrayList<ArrayList<E>> result = new ArrayList<>(matrix1.matrix.size());
        for (int i = 0; i < matrix1.matrix.size(); i++) {
            ArrayList<E> row = new ArrayList<>(matrix1.matrix.get(i).size());

            for (int j = 0; j < matrix1.matrix.get(i).size(); j++) {
                E a = matrix1.get(i, j);
                E b = matrix2.get(i, j);
                row.add(NNMatrix.addNum(a, b));
            }

            result.add(row);
        }

        return new NNMatrix<>(result);
    }

    public static <E extends Number> NNMatrix<E> Add(NNMatrix<E> matrix, NNVector<E> vector) {
        if (matrix == null || vector == null) {
            return null;
        }

        if (!Objects.equals(matrix.shape.get(1), vector.shape.get(1))) {
            throw new IllegalArgumentException(
                    "Matrices provided do not have the same number of rows and columns."
            );
        }

        ArrayList<ArrayList<E>> result = new ArrayList<>(matrix.matrix.size());
        for (int i = 0; i < matrix.matrix.size(); i++) {
            ArrayList<E> row = new ArrayList<>(matrix.matrix.get(i).size());

            for (int j = 0; j < matrix.matrix.get(i).size(); j++) {
                E a = matrix.get(i, j);
                E b = vector.get(j);
                row.add(NNMatrix.addNum(a, b));
            }

            result.add(row);
        }

        return new NNMatrix<>(result);
    }

    public static <E extends Number> NNMatrix<E> Multiply(NNMatrix<E> matrix1, NNMatrix<E> matrix2) {
        if (matrix1 == null || matrix2 == null) {
            return null;
        }

        if (!Objects.equals(matrix1.shape.get(1), matrix2.shape.get(0))) {
            throw new IllegalArgumentException(
                    "Matrices provided do not have a valid size for matrix multiplication" +
                    "(Matrix 1 columns must be equal to Matrix 2 rows)."
            );
        }

        int rows1 = matrix1.matrix.size();
        int cols1 = matrix1.matrix.get(0).size();

        int rows2 = matrix2.matrix.size();
        int cols2 = matrix2.matrix.get(0).size();

        boolean AB_valid = (cols1 == rows2);
        boolean BA_valid = (cols2 == rows1);

        if (AB_valid) {
            return NNMatrix.multiply(matrix1.matrix, matrix2.matrix, rows1, cols1, rows2, cols2);
        }

        if (BA_valid) {
            return NNMatrix.multiply(matrix2.matrix, matrix1.matrix, rows2, cols2, rows1, cols1);
        }

        throw new IllegalArgumentException(
                "NNMatrix shapes incompatible: (" + rows1 + " x " + cols1 +
                        ") cannot multiply with (" + rows2 + " x " + cols2 + ")"
        );
    }

    /*
    Helper functions
     */

    private static NNMatrix<Float> randomized(int rows, int columns, int type) {
        ArrayList<ArrayList<Float>> matrix = new ArrayList<>(rows);

        for (int i = 0; i < rows; i++) {
            ArrayList<Float> row = new ArrayList<>(columns);

            for (int j = 0; j < columns; j++) {
                switch (type) {
                    case 0 -> { // zeros
                        row.add(0f);
                    }
                    case 1 -> { // randomized
                        float val = (rng.nextFloat() * 2) - 1;
                        row.add(val);
                    }
                    case 2 -> { // HE initialization
                        float val = (float) rng.nextGaussian() * (float) Math.sqrt(2f / (float)columns);
                        row.add(val);
                    }
                }
            }

            matrix.add(row);
        }
        return new NNMatrix<>(matrix);
    }

    private static <E extends Number> E addNum(E a, E b) {
        Class<?> type = a.getClass();

        if (type == Integer.class) {
            return (E) (Integer) (a.intValue() + b.intValue());
        }
        if (type == Float.class) {
            return (E) (Float) (a.floatValue() + b.floatValue());
        }
        if (type == Double.class) {
            return (E) (Double) (a.doubleValue() + b.doubleValue());
        }
        if (type == Long.class) {
            return (E) (Long) (a.longValue() + b.longValue());
        }
        if (type == Short.class) {
            return (E) (Short) (short) (a.shortValue() + b.shortValue());
        }
        if (type == Byte.class) {
            return (E) (Byte) (byte) (a.byteValue() + b.byteValue());
        }

        throw new IllegalArgumentException("Unsupported numeric type: " + type.getName());
    }

    private static <E extends Number> NNMatrix<E> multiply(
            ArrayList<ArrayList<E>> matrix1,
            ArrayList<ArrayList<E>> matrix2,
            int rowsL, int colsL,
            int rowsR, int colsR)
    {
        ArrayList<ArrayList<E>> result = new ArrayList<>(rowsL);
        Class<?> type = matrix1.get(0).get(0).getClass();

        for (int i = 0; i < rowsL; i++) {
            ArrayList<E> row = new ArrayList<>();
            for (int j = 0; j < colsR; j++) {
                if (type == Integer.class) {
                    row.add((E) (Integer) 0);
                }
                if (type == Float.class) {
                    row.add((E) (Float) 0f);
                }
                if (type == Double.class) {
                    row.add((E) (Double) 0d);
                }
                if (type == Long.class) {
                    row.add((E) (Long) 0L);
                }
                if (type == Short.class) {
                    row.add((E) (Short) (short) 0);
                }
                if (type == Byte.class) {
                    row.add((E) (Byte) (byte) 0);
                }
            }
            result.add(row);
        }

        for (int i = 0; i < rowsL; i++) {
            for (int j = 0; j < colsR; j++) {
                Number sum = 0;
                for (int k = 0; k < colsL; k++) {
                    if (type == Integer.class) {
                        sum = sum.intValue() + ((matrix1.get(i).get(k)).intValue() * matrix2.get(k).get(j).intValue());
                    }
                    if (type == Float.class) {
                        sum = sum.floatValue() + ((matrix1.get(i).get(k)).floatValue() * matrix2.get(k).get(j).floatValue());
                    }
                    if (type == Double.class) {
                        sum = sum.doubleValue() + ((matrix1.get(i).get(k)).doubleValue() * matrix2.get(k).get(j).doubleValue());
                    }
                    if (type == Long.class) {
                        sum = sum.longValue() + ((matrix1.get(i).get(k)).longValue() * matrix2.get(k).get(j).longValue());
                    }
                    if (type == Short.class) {
                        sum = sum.shortValue() + ((matrix1.get(i).get(k)).shortValue() * matrix2.get(k).get(j).shortValue());
                    }
                    if (type == Byte.class) {
                        sum = sum.byteValue() + ((matrix1.get(i).get(k)).byteValue() * matrix2.get(k).get(j).byteValue());
                    }
                }
                result.get(i).set(j, (E) sum);
            }
        }

        return new NNMatrix<>(result);
    }
}
