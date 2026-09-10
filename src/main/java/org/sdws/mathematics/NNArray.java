package org.sdws.mathematics;

import java.util.*;

/**
 * NNArray class
 */
public class NNArray {
    private ArrayList<?> nnarray;
    public int elements, dimension;
    public ArrayList<Integer> shape;
    private static Random rand = new Random();

    private NNArray(ArrayList<?> array) {
        if (Util.validInput(array)) {
            this.nnarray = new ArrayList<>(array);
            this.elements = this.getElements(this.nnarray);
            this.shape = this.getShape(this.nnarray);
            this.dimension = this.shape.size();

            if (this.dimension == 2 && this.shape.get(0) == 1) {
                this.elements = this.getElements(this.nnarray);
                this.nnarray = this.flatten(this.nnarray);
                this.shape = this.getShape(this.nnarray);
                this.dimension = this.shape.size();
            }
        } else {
            throw new IllegalArgumentException("NNArray input is invalid for matrix.");
        }
    }

    private void refresh(ArrayList<?> array) {
        if (Util.validInput(array)) {
            this.nnarray = new ArrayList<>(array);
            this.elements = this.getElements(this.nnarray);
            this.shape = this.getShape(this.nnarray);
            this.dimension = this.shape.size();
        }
    }

    /**
     * Recursively proceeds into the matrix input of type {@link ArrayList}{@code <?>} and
     * summates the amount of {@link Float} elements.
     * @param input either {@link Float} as base case, or {@link ArrayList}{@code <?>}
     * @return a single float value of the list type element
     */
    private Integer getElements(Object input) {
        if (input instanceof Float) {
            return 1;
        } else {
            int elements = 0;

            for (Object element : (List<?>) input) {
                elements += this.getElements(element);
            }

            return elements;
        }
    }

    /**
     * Recursively proceeds into the matrix input of type {@link ArrayList}{@code <?>} and
     * appends returned values in sub-lists (if any) to return a total dimension
     * of the overall matrix argument.
     * @param input either {@link Float} as base case, or {@link ArrayList}{@code <?>}
     * @return a list of integers corresponding to the dimension of the ArrayList matrix
     */
    private ArrayList<Integer> getShape(Object input) {
        if (input instanceof Float) {
            return new ArrayList<>();
        } else {
            ArrayList<?> list = (ArrayList<?>) input;

            ArrayList<Integer> dimension = new ArrayList<>();
            dimension.add(list.size());

            if (list.isEmpty()) {
                return dimension;
            }

            List<Integer> subListDim = getShape(list.get(0));
            dimension.addAll(subListDim);

            return dimension;
        }
    }

    /**
     * Flattens a <code>1</code> x <code>N</code> input, where <code>N</code> is
     * any real number, into a {@link NNArray} object representing a vector (a list of depth 1).
     */
    private ArrayList<Float> flatten(Object input) {
        if (input instanceof Float f) {
            ArrayList<Float> list = new ArrayList<>();
            list.add(f);
            return list;
        } else {
            ArrayList<?> list = (ArrayList<?>) input;

            ArrayList<Float> result = new ArrayList<>();
            for (int i = 0; i < list.size(); i++) {
                result.addAll(this.flatten(list.get(i)));
            }

            return result;
        }
    }

    /**
     * Returns the internal array attribute for this object.
     * @return a {@link NNArray} object represented by type {@link ArrayList}
     */
    public ArrayList<?> nnarray() {
        return this.nnarray;
    }

    public static NNArray add(NNArray curr, NNArray other) {
        if (curr.dimension != other.dimension) {
            if ((curr.dimension == 2 && other.dimension == 1) || (curr.dimension == 1 && other.dimension == 2)) {
                return Util.matrixAddVector(curr, other);
            }
        } else {
            if (!Objects.equals(curr.shape, other.shape)) {
                throw new IllegalArgumentException("Shape(s) invalid for operation.");
            } else {
                if (curr.dimension == 1) {
                    return Util.vectorAddVector(curr, other);
                } else if (curr.dimension == 2) {
                    return Util.matrixAddMatrix(curr, other);
                } else {
                    throw new IllegalArgumentException("N/A.");
                }
            }
        }

        throw new IllegalArgumentException("NNArray input is invalid for matrix.");
    }

    public Float get(int index) {
        if (this.dimension == 1) {
            if (index > this.elements - 1) {
                throw new IndexOutOfBoundsException("Index out of bounds.");
            } else {
                return (Float) this.nnarray.get(index);
            }
        } else {
            throw new IllegalArgumentException("Invalid operation on this NNArray.");
        }
    }

    public void set(int index, Float value) {
        if (this.dimension == 1) {
            if (index > this.elements - 1) {
                throw new IndexOutOfBoundsException("Index out of bounds.");
            } else {
                ((ArrayList<Float>)this.nnarray).set(index, value);
            }
        } else {
            throw new IllegalArgumentException("Invalid operation on this NNArray.");
        }
    }

    public Float get(int row, int column) {
        if (this.dimension == 2) {
            if ((row <= this.shape.get(0) - 1) && (column <= this.shape.get(1) - 1)) {
                ArrayList<Float> rowList = (ArrayList<Float>) this.nnarray.get(row);
                return rowList.get(column);
            } else {
                throw new IndexOutOfBoundsException("Row/column index out of bounds.");
            }
        } else {
            throw new IllegalArgumentException("Invalid operation on this NNArray.");
        }
    }

    public void set(int row, int column, Float value) {
        if (this.dimension == 2) {
            if ((row <= this.shape.get(0) - 1) && (column <= this.shape.get(1) - 1)) {
                ArrayList<Float> rowList = (ArrayList<Float>) this.nnarray.get(row);
                rowList.set(column, value);
            } else {
                throw new IndexOutOfBoundsException("Row/column index out of bounds.");
            }
        } else {
            throw new IllegalArgumentException("Invalid operation on this NNArray.");
        }
    }

    public Float floatOutput() {
        if (this.dimension > 1) {
            throw new IllegalArgumentException("Cannot derive a float output from a matrix.");
        }

        if (this.elements > 1) {
            throw new IllegalArgumentException("Cannot derive a float output from a vector.");
        }

        return this.get(0);
    }

    public boolean isFloat() {
        return this.dimension == 1 && this.elements == 1;
    }

    public NNArray deepCopy() {
        switch (this.dimension) {
            case 1 -> {
                ArrayList<Float> copy = new ArrayList<>();

                for (int i = 0; i < this.nnarray.size(); i++) {
                    copy.add(((ArrayList<Float>)this.nnarray).get(i));
                }

                return NNArray.create(copy);
            }
            case 2 -> {
                ArrayList<ArrayList<Float>> copy = new ArrayList<>();

                for (int i = 0; i < this.nnarray.size(); i++) {
                    ArrayList<Float> rowList = (ArrayList<Float>)this.nnarray.get(i);
                    ArrayList<Float> copyRow = new ArrayList<>();

                    for (Float element : rowList) {
                        copyRow.add(element);
                    }
                    copy.add(copyRow);
                }

                return NNArray.create(copy);
            }
            default -> {
                throw new IllegalArgumentException("Invalid operation on this NNArray.");
            }
        }
    }

    /**
     * Transpose the current {@link NNArray} via object mutation.
     * Can also transpose 1-dimensional vectors into a multirow matrix
     * with a single column.
     */
    public void T() {
        if (this.dimension < 2) { // transpose vector -> matrix
            ArrayList<ArrayList<?>> column = new ArrayList<>();

            for (int i = 0; i < this.shape.get(0); i++) {
                ArrayList<Float> row = new ArrayList<>();
                row.add(this.get(i));
                column.add(row);
            }

            this.refresh(column);
        } else { // transpose matrix -> matrix/vector
            if (this.shape.get(1) <= 1) { // transpose into a vector
                ArrayList<Float> vec = new ArrayList<>();
                ArrayList<ArrayList<Float>> curr = (ArrayList<ArrayList<Float>>) this.nnarray;

                for  (int i = 0; i < curr.size(); i++) {
                    ArrayList<Float> row = curr.get(i);
                    vec.add(row.get(0));
                }

                this.refresh(vec);
            } else { // regular matrix transpose algorithm
                ArrayList<ArrayList<Float>> transposed = new ArrayList<>();

                int newRows = this.shape.get(1);
                int newColumns = this.shape.get(0);

                for (int i = 0; i < newRows; i++) {
                    ArrayList<Float> row = new ArrayList<>();
                    for (int j = 0; j < newColumns; j++) {
                        row.add(0f);
                    }

                    transposed.add(row);
                }

                for (int i = 0; i < this.shape.get(0); i++) {
                    for (int j = 0; j < this.shape.get(1); j++) {
                        transposed.get(j).set(i, this.get(i, j));
                    }
                }

                this.refresh(transposed);
            }
        }
    }

    @Override
    public String toString() {
        return this.nnarray.toString();
    }

    /**
     * Creation of an n-dimensional array.
     * @param input multidimensional float array of type {@link List}
     * @return a {@link NNArray} object
     */
    public static NNArray create(List<?> input) {
        return new NNArray((ArrayList<?>) Util.toArrayList(input));
    }

    /**
     * Create a {@link NNArray} object primarily for vector based arrays.
     * @param input 1-dimensional float array of type {@code float[]}
     * @return a {@link NNArray} object
     */
    public static NNArray create(float[] input) {
        ArrayList<Float> array = new ArrayList<>();
        for (float element : input) {
            array.add(element);
        }

        return new NNArray(array);
    }

    /**
     * Create a {@link NNArray} object primarily for vector based arrays, with
     * all values from indexes 0 - (<code>capacity</code> - 1) being 0.
     * @param capacity the length of the vector
     * @return a {@link NNArray} object
     */
    public static NNArray createZeros(int capacity) {
        float[] vec = new float[capacity];

        for (int i = 0; i < capacity; i++) {
            vec[i] = 0f;
        }

        return NNArray.create(vec);
    }

    /**
     * Create a {@link NNArray} object primarily for vector based arrays, with
     * randomized values.
     * @param capacity the length of the vector
     * @return a {@link NNArray} object
     */
    public static NNArray createRandom(int capacity) {
        float[] vec = new float[capacity];

        for (int i = 0; i < capacity; i++) {
            float val = (rand.nextFloat() * 2) - 1;
            val *= 1000;
            vec[i] = Math.round(val) / 1000f;
        }

        return NNArray.create(vec);
    }

    /**
     * Create a {@link NNArray} object primarily for vector based arrays, with
     * values computed via the He Initialization algorithm, which leverages
     * gaussian distribution. This is particularly useful for vectors involved in
     * layers that use the ReLU activation function.
     * @param capacity the length of the vector
     * @return a {@link NNArray} object
     */
    public static NNArray HeInitialization(int capacity) {
        float[] vec = new float[capacity];

        for (int i = 0; i < capacity; i++) {
            float val = (float) rand.nextGaussian() * (float)Math.sqrt(2f / (float)capacity);
            vec[i] = Math.round(val * 1000) / 1000f;
            //vec[i] = val;
        }

        return NNArray.create(vec);
    }

    public static NNArray createZeros(int rows, int columns) {
        List<List<Float>> outer = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Float> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                row.add(0f);
            }

            outer.add(row);
        }

        return NNArray.create(outer);
    }

    public static NNArray createRandom(int rows, int columns) {
        List<List<Float>> outer = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Float> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                float val = (rand.nextFloat() * 2) - 1;
                val *= 1000;
                row.add(Math.round(val) / 1000f);
            }

            outer.add(row);
        }

        return NNArray.create(outer);
    }

    public static NNArray HeInitialization(int rows, int columns) {
        List<List<Float>> outer = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<Float> row = new ArrayList<>();
            for (int j = 0; j < columns; j++) {
                float val = (float) rand.nextGaussian() * (float)Math.sqrt(2f / (float)columns);
                row.add(Math.round(val * 1000) / 1000f);
                //row.add(val);
            }

            outer.add(row);
        }

        return NNArray.create(outer);
    }

    public static NNArray dot(NNArray A, NNArray B) {
        ArrayList<?> objA = A.nnarray();
        ArrayList<?> objB = B.nnarray();

        if (A.dimension == 1 && B.dimension == 2) {
            return Util.matrixVectorProduct(objA, objB);
        } else if (A.dimension == 2 && B.dimension == 1) {
            return Util.matrixVectorProduct(objB, objA);
        } else if (A.dimension == 2 && B.dimension == 2) {
            return Util.matrixProduct(objA, objB);
        } else if (A.dimension == 1 && B.dimension == 1) {
            return Util.dotProduct(objA, objB);
        }

        throw new IllegalArgumentException("Unsupported NNArray dot operation.");
    }

    /**
     * Private utilities class with extra computational methods to be
     * used only within {@link NNArray} class methods/functions.
     */
    private static class Util {
        /**
         * Recursively determines if the passed argument, usually of type {@link List}{@code <?>}
         * is properly structured to be initialized into an NNArray object.
         * @param input either type {@link Float} for base case or {@link List}{@code <?>}
         * @return a boolean value determining if the overall passed argument
         * is valid for {@link NNArray} creation
         */
        public static boolean validInput(Object input) {
            if (input instanceof Float) {
                return true;
            }

            if (!(input instanceof List<?> list)) {
                return false;
            }

            if (list.isEmpty()) {
                return true;
            }

            Object first = list.get(0);

            if (first instanceof Float) {
                for (Object o : list) {
                    if (!(o instanceof Float)) {
                        return false;
                    }
                }
                return true;
            }

            if (first instanceof List<?> firstList) {
                int startSize = firstList.size();

                for (Object o : list) {
                    if (!(o instanceof List<?> subList)) {
                        return false;
                    }
                    if (!validInput(o)) {
                        return false;
                    }
                    if (subList.size() != startSize) {
                        return false;
                    }
                }
                return true;
            }

            return false;
        }

        /**
         * Recursively creates a deep-copy of passed argument 'List<?>'.
         * @param input either an {@link Object} value as base case, or {@link ArrayList}{@code <?>}
         * @return {@link ArrayList}{@code <?>} deep-copy
         */
        public static Object toArrayList(Object input) {
            if (!(input instanceof List<?>)) {
                return input;
            }

            List<?> original = (List<?>) input;
            ArrayList<Object> copy = new ArrayList<>();

            for (Object element : original) {
                if (element instanceof List<?>) {
                    copy.add(toArrayList(element));
                } else {
                    copy.add(element);
                }
            }

            return copy;
        }

        public static NNArray dotProduct(Object A, Object B) {
            ArrayList<?> listA = (ArrayList<?>) A;
            ArrayList<?> listB = (ArrayList<?>) B;

            if (listA.size() != listB.size()) {
                throw new IllegalArgumentException("Vectors are not the same size for dot product.");
            }

            float sum = 0;

            for (int i = 0; i < listB.size(); i++) {
                sum += (Float) listA.get(i) * (Float) listB.get(i);
            }

            return NNArray.create(new float[]{sum});
        }

        public static NNArray matrixVectorProduct(Object A, Object B) {
            ArrayList<?> mat = (ArrayList<?>) B;
            ArrayList<?> vec = (ArrayList<?>) A;

            int rows = mat.size();
            int cols = ((ArrayList<?>) mat.get(0)).size();

            ArrayList<Float> result = new ArrayList<>();

            for (int i = 0; i < rows; i++) {
                float sum = 0f;
                for (int j = 0; j < cols; j++) {
                    sum += (Float) ((ArrayList<?>) mat.get(i)).get(j) * (Float) vec.get(j);
                }
                result.add(sum);
            }

            return new NNArray(result);
        }

        public static NNArray matrixProduct(Object A, Object B) {
            ArrayList<?> listA = (ArrayList<?>) A;
            ArrayList<?> listB = (ArrayList<?>) B;

            int rowsA = listA.size();
            int colsA = ((ArrayList<?>) listA.get(0)).size();

            int rowsB = listB.size();
            int colsB = ((ArrayList<?>) listB.get(0)).size();

            boolean AB_valid = (colsA == rowsB);
            boolean BA_valid = (colsB == rowsA);

            if (AB_valid) {
                return multiply(listA, listB, rowsA, colsA, rowsB, colsB);
            }

            if (BA_valid) {
                return multiply(listB, listA, rowsB, colsB, rowsA, colsA);
            }

            throw new IllegalArgumentException(
                    "NNMatrix shapes incompatible: (" + rowsA + " x " + colsA +
                            ") cannot multiply with (" + rowsB + " x " + colsB + ")"
            );
        }

        private static NNArray multiply(
                ArrayList<?> left,
                ArrayList<?> right,
                int rowsL, int colsL,
                int rowsR, int colsR) {

            ArrayList<ArrayList<Float>> result = new ArrayList<>(rowsL);

            for (int i = 0; i < rowsL; i++) {
                ArrayList<Float> row = new ArrayList<>();
                for (int j = 0; j < colsR; j++) {
                    row.add(0f);
                }
                result.add(row);
            }

            for (int i = 0; i < rowsL; i++) {
                for (int j = 0; j < colsR; j++) {
                    float sum = 0f;
                    for (int k = 0; k < colsL; k++) {
                        float a = (Float) ((ArrayList<?>) left.get(i)).get(k);
                        float b = (Float) ((ArrayList<?>) right.get(k)).get(j);
                        sum += a * b;
                    }
                    result.get(i).set(j, sum);
                }
            }

            return new NNArray(result);
        }


        public static NNArray vectorAddVector(NNArray A, NNArray B) {
            ArrayList<Float> result = new ArrayList<>();

            if (!Objects.equals(A.shape.get(0), B.shape.get(0))) {
                throw new IllegalArgumentException("NNArray vector sizes do not match.");
            }

            for (int i = 0; i < A.shape.get(0); i++) {
                result.add(A.get(i) + B.get(i));
            }

            return new NNArray(result);
        }

        public static NNArray matrixAddVector(NNArray A, NNArray B) {
            int maxRows, maxColumns;
            NNArray currentMatrixRef, currentVectorRef;
            if (A.dimension >  B.dimension) {   // A is a matrix, B is a vector
                maxRows = A.shape.get(0);
                maxColumns = A.shape.get(1);
                currentMatrixRef = A;
                currentVectorRef = B;
            } else {    // A is a vector, B is a matrix
                maxRows = B.shape.get(0);
                maxColumns = B.shape.get(1);
                currentMatrixRef = B;
                currentVectorRef = A;
            }

            if (!Objects.equals(A.shape.get(0), B.shape.get(0))) {
                throw new IllegalArgumentException("NNArray vector and matrix sizes do not match.");
            }

            ArrayList<ArrayList<Float>> result = new ArrayList<>();

            for (int i = 0; i < maxRows; i++) {
                ArrayList<Float> row = new ArrayList<>();
                for (int j = 0; j < maxColumns; j++) {
                    row.add(currentMatrixRef.get(i, j));
                }

                result.add(row);
            }

            for (int row = 0; row < result.size(); row++) {
                for (int col = 0; col < result.get(row).size(); col++) {
                    float val = result.get(row).get(col) + currentVectorRef.get(row);
                    result.get(row).set(col, Math.round(val * 1000) / 1000f);
                }
            }

            return new NNArray(result);
        }

        public static NNArray matrixAddMatrix(NNArray A, NNArray B) {
            if (!Objects.equals(A.shape, B.shape)) {
                throw new IllegalArgumentException("NNArray matrix sizes do not match.");
            }

            ArrayList<ArrayList<Float>> result = new ArrayList<>();

            for (int i = 0; i < A.shape.get(0); i++) {
                ArrayList<Float> row = new ArrayList<>();
                for (int j = 0; j < A.shape.get(1); j++) {
                    row.add(A.get(i, j));
                }

                result.add(row);
            }

            for (int row = 0; row < result.size(); row++) {
                for (int col = 0; col < result.get(row).size(); col++) {
                    float val = result.get(row).get(col) + B.get(row, col);
                    result.get(row).set(col, Math.round(val * 1000) / 1000f);
                }
            }

            return new NNArray(result);
        }
    }
}
