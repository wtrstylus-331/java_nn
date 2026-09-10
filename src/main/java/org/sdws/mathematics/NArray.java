package org.sdws.mathematics;

import java.util.*;

public class NArray<E extends Number> {
    private ArrayList<?> array;
    public int elements, dimension;
    public ArrayList<Integer> shape;
    private static Random rand = new Random();

    /*
    Class methods
     */

    private NArray(ArrayList<?> array) {
        this.array = array;
        this.dimension = this.getDimension(this.array);
        this.elements = this.getElements(this.array);
        this.shape = new ArrayList<>(this.dimension);

        if (this.dimension == 1) {
            this.shape.add(1);
            this.shape.add(this.array.size());
        } else {
            this.shape.add(this.array.size());
            this.shape.add(((ArrayList<ArrayList<E>>)this.array).get(0).size());
        }
    }

    @Override
    public String toString() {
        return this.array.toString();
    }

    private Integer getElements(Object input) {
        if (!(input instanceof ArrayList)) {
            return 1;
        } else {
            int elements = 0;

            for (Object sublist : (ArrayList<?>) input) {
                elements += this.getElements(sublist);
            }

            return elements;
        }
    }

    private Integer getDimension(Object input) {
        if (!(input instanceof ArrayList)) {
            return 0;
        } else {
            int dim = 1;
            dim += this.getDimension(((ArrayList<?>) input).get(0));
            return dim;
        }
    }

    public E get(int index) {
        if (this.array.isEmpty()) {
            return null;
        }

        if (this.dimension > 1) {
            throw new IllegalArgumentException("Cannot retrieve an element using this variation of the class method.");
        }

        if (index >= this.array.size() || index < 0) {
            throw new IndexOutOfBoundsException("Index: " + index + " is out of bounds for this NArray.");
        }

        return ((ArrayList<E>)this.array).get(index);
    }

    public void set(int index, E element) {
        if (!this.array.isEmpty()) {
            if (this.dimension > 1) {
                throw new IllegalArgumentException("Cannot retrieve an element using this variation of the class method.");
            }

            if (index >= this.array.size() || index < 0) {
                throw new IndexOutOfBoundsException("Index: " + index + " is out of bounds for this NArray.");
            }

            ((ArrayList<E>)this.array).set(index, element);
        } else {
            throw new IllegalArgumentException("Cannot set an element into an empty NArray.");
        }
    }

    public E get(int row, int column) {
        if (this.array.isEmpty()) {
            return null;
        }

        if (this.dimension < 2) {
            throw new IllegalArgumentException("Cannot retrieve an element using this variation of the class method.");
        }

        if (row >= this.array.size() || row < 0) {
            throw new IndexOutOfBoundsException("Row index: " + row + " is out of bounds for this NArray.");
        }

        if (column >= ((ArrayList<E>)this.array.get(row)).size() || column < 0) {
            throw new IndexOutOfBoundsException("Column index: " + column + " is out of bounds for this NArray.");
        }

        return ((ArrayList<ArrayList<E>>)this.array).get(row).get(column);
    }

    public void set(int row, int column, E element) {
        if (!this.array.isEmpty()) {
            if (this.dimension < 2) {
                throw new IllegalArgumentException("Cannot retrieve an element using this variation of the class method.");
            }

            if (row >= this.array.size() || row < 0) {
                throw new IndexOutOfBoundsException("Row index: " + row + " is out of bounds for this NArray.");
            }

            if (column >= ((ArrayList<E>)this.array.get(row)).size() || column < 0) {
                throw new IndexOutOfBoundsException("Column index: " + column + " is out of bounds for this NArray.");
            }

            ((ArrayList<ArrayList<E>>)this.array).get(row).set(column, element);
        } else {
            throw new IllegalArgumentException("Cannot set an element into an empty NArray.");
        }
    }

    public ArrayList<?> innerArray() {
        return this.array;
    }

    public void transpose() {
        if (this.dimension < 2) { // transpose vector -> matrix
            ArrayList<ArrayList<E>> column = new ArrayList<>();

            for (int i = 0; i < this.shape.get(1); i++) {
                ArrayList<E> row = new ArrayList<>();
                row.add(this.get(i));
                column.add(row);
            }

            this.array = column;
        } else { // transpose matrix -> matrix/vector
            if (this.shape.get(1) <= 1) { // transpose into a vector
                ArrayList<E> vec = new ArrayList<>();
                ArrayList<ArrayList<E>> curr = (ArrayList<ArrayList<E>>) this.array;

                for  (int i = 0; i < curr.size(); i++) {
                    ArrayList<E> row = curr.get(i);
                    vec.add(row.get(0));
                }

                this.array = vec;
            } else { // regular matrix transpose algorithm
                ArrayList<ArrayList<E>> transposed = new ArrayList<>();

                int newRows = this.shape.get(1);
                int newColumns = this.shape.get(0);

                for (int i = 0; i < newRows; i++) {
                    ArrayList<E> row = new ArrayList<>();
                    for (int j = 0; j < newColumns; j++) {
                        if (this.get(0,0).getClass().equals(Float.class)) {
                            row.add((E) (Float) 0f);
                        } else if (this.get(0,0).getClass().equals(Double.class)) {
                            row.add((E) (Double) 0d);
                        } else if (this.get(0,0).getClass().equals(Integer.class)) {
                            row.add((E) (Integer) 0);
                        } else if (this.get(0,0).getClass().equals(Long.class)) {
                            row.add((E) (Long) 0L);
                        } else if (this.get(0,0).getClass().equals(Short.class)) {
                            row.add((E) (Short) (short) 0);
                        } else if (this.get(0,0).getClass().equals(Byte.class)) {
                            row.add((E) (Byte) (byte) 0);
                        } else {
                            row.add(null);
                        }
                    }

                    transposed.add(row);
                }

                for (int i = 0; i < this.shape.get(0); i++) {
                    for (int j = 0; j < this.shape.get(1); j++) {
                        transposed.get(j).set(i, this.get(i, j));
                    }
                }

                this.array = transposed;
            }
        }

        this.dimension = this.getDimension(this.array);
        this.shape.clear();

        if (this.dimension == 1) {
            this.shape.add(1);
            this.shape.add(this.array.size());
        } else {
            this.shape.add(this.array.size());
            this.shape.add(((ArrayList<ArrayList<E>>)this.array).get(0).size());
        }
    }

    /*
    Static class functions
     */

    public static <E extends Number> NArray<E> FromRows(List<E>... rows) {
        if (rows.length == 0 || rows[0].isEmpty()) {
            return new NArray<>(new ArrayList<>(0));
        }

        if (rows.length == 1) { // create a vector instead since it is essentially 1 row
            return NArray.FromCollection(rows[0]);
        }

        ArrayList<ArrayList<E>> matrix = new ArrayList<>();
        for (List<E> row : rows) {
            if (!Objects.equals(row.size(), rows[0].size())) {
                throw new IllegalArgumentException(
                        "Input list rows must have the same number of columns for matrix initialization."
                );
            }

            matrix.add(new ArrayList<>(row));
        }
        return new NArray<>(matrix);
    }

    public static <E extends Number> NArray<E> FromNested(ArrayList<ArrayList<E>> arrayList) {
        if (arrayList.isEmpty()) {
            return null;
        }

        if (arrayList.size() < 2) {
            return NArray.FromCollection(arrayList.get(0));
        }

        return new NArray<>(arrayList);
    }

    public static <E extends Number> NArray<E> FromElements(E... elements) {
        if (elements.length == 0) {
            return new NArray<>(new ArrayList<>());
        }

        return new NArray<>(new ArrayList<>(List.of(elements)));
    }

    public static <E extends Number> NArray<E> FromCollection(Collection<E> collection) {
        if (collection.isEmpty()) {
            return new NArray<>(new ArrayList<>());
        }

        return new NArray<>(new ArrayList<>(collection));
    }

    public static <E extends Number> NArray<E> CreateZeros(Class<E> type, int length) {
        ArrayList<E> vec = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            if (type.equals(Float.class)) {
                vec.add((E) (Float) 0f);
            } else if (type.equals(Double.class)) {
                vec.add((E) (Double) 0d);
            } else if (type.equals(Integer.class)) {
                vec.add((E) (Integer) 0);
            } else if (type.equals(Long.class)) {
                vec.add((E) (Long) 0L);
            } else if (type.equals(Short.class)) {
                vec.add((E) (Short) (short) 0);
            } else if (type.equals(Byte.class)) {
                vec.add((E) (Byte) (byte) 0);
            }
        }

        return NArray.FromCollection(vec);
    }

    public static <E extends Number> NArray<E> CreateZeros(int length) {
        ArrayList<E> vec = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            vec.add((E) (Float) 0f);
        }

        return NArray.FromCollection(vec);
    }

    public static <E extends Number> NArray<E> CreateZeros(Class<E> type, int rows, int columns) {
        ArrayList<ArrayList<E>> matrix = new ArrayList<>(rows);

        for (int i = 0; i < rows; i++) {
            ArrayList<E> row = new ArrayList<>(columns);
            for (int j = 0; j < columns; j++) {
                if (type.equals(Float.class)) {
                    row.add((E) (Float) 0f);
                } else if (type.equals(Double.class)) {
                    row.add((E) (Double) 0d);
                } else if (type.equals(Integer.class)) {
                    row.add((E) (Integer) 0);
                } else if (type.equals(Long.class)) {
                    row.add((E) (Long) 0L);
                } else if (type.equals(Short.class)) {
                    row.add((E) (Short) (short) 0);
                } else if (type.equals(Byte.class)) {
                    row.add((E) (Byte) (byte) 0);
                } else {
                    row.add((E) (Float) 0f);
                }
            }
            matrix.add(row);
        }

        return NArray.FromNested(matrix);
    }

    public static <E extends Number> NArray<E> CreateZeros(int rows, int columns) {
        ArrayList<ArrayList<E>> matrix = new ArrayList<>(rows);

        for (int i = 0; i < rows; i++) {
            ArrayList<E> row = new ArrayList<>(columns);
            for (int j = 0; j < columns; j++) {
                row.add((E) (Float) 0f);
            }
            matrix.add(row);
        }

        return NArray.FromNested(matrix);
    }

    public static <E extends Number> NArray<E> CreateRandom(int length) {
        ArrayList<E> vec = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            vec.add((E) (Float) ((rand.nextFloat() * 2) - 1));
        }

        return NArray.FromCollection(vec);
    }

    public static <E extends Number> NArray<E> CreateRandom(int rows, int columns) {
        ArrayList<ArrayList<E>> matrix = new ArrayList<>(rows);

        for (int i = 0; i < rows; i++) {
            ArrayList<E> row = new ArrayList<>(columns);
            for (int j = 0; j < columns; j++) {
                row.add((E) (Float) ((rand.nextFloat() * 2) - 1));
            }
            matrix.add(row);
        }

        return NArray.FromNested(matrix);
    }

//    public static <E extends Number> NArray<E> InitializeHe(int length) {
//        if (collection.isEmpty()) {
//            return new NArray<>(new ArrayList<>());
//        }
//
//        return new NArray<>(new ArrayList<>(collection));
//    }

    public static <E extends Number> NArray<E> InitializeHe(int rows, int columns) {
        ArrayList<ArrayList<E>> matrix = new ArrayList<>(rows);

        for (int i = 0; i < rows; i++) {
            ArrayList<E> row = new ArrayList<>(columns);
            for (int j = 0; j < columns; j++) {
                row.add((E) (Float) ((float) rand.nextGaussian() * (float)Math.sqrt(2f / (float)columns)));
            }
            matrix.add(row);
        }

        return NArray.FromNested(matrix);
    }

    public static <E extends Number> NArray<E> DeepCopy(NArray<E> array) {
        if (array == null) {
            return null;
        }

        if (array.dimension == 1) { // copy a vector
            ArrayList<E> list = new ArrayList<>();

            for (E elem : (ArrayList<E>) array.array) {
                list.add(elem);
            }

            return new NArray<>(list);
        } else { // copy a matrix
            ArrayList<ArrayList<E>> result = new ArrayList<>(array.shape.get(0));

            for (int i = 0; i < array.shape.get(0); i++) {
                ArrayList<E> origRow = (ArrayList<E>) array.array.get(i);
                ArrayList<E> copyRow = new ArrayList<>(((ArrayList<E>) array.array.get(i)).size());

                for (E element : origRow) {
                    copyRow.add(element);
                }

                result.add(copyRow);
            }

            return new NArray<>(result);
        }
    }

    public static <E extends Number> NArray<E> Add(NArray<E> array1, NArray<E> array2) {
        if (array1 == null ||  array2 == null) {
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

    public static <E extends Number> NArray<E> Subtract(NArray<E> array1, NArray<E> array2) {
        if (array1 == null ||  array2 == null) {
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

    public static <E extends Number> E Dot(NArray<E> array1, NArray<E> array2) {
        if (array1 == null ||  array2 == null) {
            return null;
        }

        if (array1.dimension == array2.dimension) {
            if (array1.dimension == 1) {
                return NArray.dotProduct(array1, array2);
            }
        }

        throw new IllegalArgumentException("Cannot determine dot product with provided NArray arguments.");
    }

    public static <E extends Number> NArray<E> Mul(NArray<E> array1, NArray<E> array2) {
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
            return NArray.matrixVecProduct(array1, array2);
        }
    }

    /*
    Private static helper functions
     */

    private static <E extends Number> NArray<E> operateVecs(NArray<E> vec1, NArray<E> vec2, boolean addition) {
        if (!Objects.equals(vec1.shape, vec2.shape)) {
            throw new IllegalArgumentException("Cannot add vector arrays that do not have the same number of elements.");
        }

        ArrayList<E> result = new ArrayList<>(vec1.shape.get(1));

        for (int i = 0; i < vec1.shape.get(1); i++) {
            if (addition) {
                result.add(NArray.addNums(vec1.get(i), vec2.get(i)));
            } else {
                result.add(NArray.subtractNums(vec1.get(i), vec2.get(i)));
            }
        }

        return new NArray<>(result);
    }

    private static <E extends Number> NArray<E> operateMatrices(NArray<E> matrix1, NArray<E> matrix2, boolean addition) {
        if (!Objects.equals(matrix1.shape, matrix2.shape)) {
            throw new IllegalArgumentException("Cannot add matrix arrays that do not have the same shape.");
        }

        ArrayList<ArrayList<E>> result = new ArrayList<>(matrix1.shape.get(0));
        for (int i = 0; i < matrix1.shape.get(0); i++) {
            ArrayList<E> row = new ArrayList<>(matrix1.shape.get(1));

            for (int j = 0; j < matrix1.shape.get(1); j++) {
                E a = matrix1.get(i, j);
                E b = matrix2.get(i, j);

                if (addition) {
                    row.add(NArray.addNums(a, b));
                } else {
                    row.add(NArray.subtractNums(a, b));
                }
            }

            result.add(row);
        }

        return new NArray<>(result);
    }

    private static <E extends Number> NArray<E> operMatrixVec(NArray<E> array1, NArray<E> array2, boolean addition) {
        if (!Objects.equals(array1.shape.get(1), array2.shape.get(1))) {
            throw new IllegalArgumentException(
                    "Vector length does not match number of matrix columns."
            );
        }

        if (array1.shape.get(0) > 1) { // first arg is a matrix, second arg is a vector
            ArrayList<ArrayList<E>> result = new ArrayList<>(array1.shape.get(0));
            for (int i = 0; i < array1.shape.get(0); i++) {
                ArrayList<E> row = new ArrayList<>(array1.shape.get(1));

                for (int j = 0; j < array1.shape.get(1); j++) {
                    E a = array1.get(i, j);
                    E b = array2.get(j);

                    if (addition) {
                        row.add(NArray.addNums(a, b));
                    } else {
                        row.add(NArray.subtractNums(a, b));
                    }
                }

                result.add(row);
            }

            return new NArray<>(result);
        } else { // first arg is a vector, second arg is a matrix
            ArrayList<ArrayList<E>> result = new ArrayList<>(array2.shape.get(0));
            for (int i = 0; i < array2.shape.get(0); i++) {
                ArrayList<E> row = new ArrayList<>(array2.shape.get(1));

                for (int j = 0; j < array2.shape.get(1); j++) {
                    E a = array2.get(i, j);
                    E b = array1.get(j);

                    if (addition) {
                        row.add(NArray.addNums(a, b));
                    } else {
                        row.add(NArray.subtractNums(a, b));
                    }
                }

                result.add(row);
            }

            return new NArray<>(result);
        }
    }

    private static <E extends Number> E dotProduct(NArray<E> vec1, NArray<E> vec2) {
        if (!Objects.equals(vec1.shape, vec2.shape)) {
            throw new IllegalArgumentException("Cannot compute dot product of vectors of different length.");
        }

        if (Objects.equals(vec1.shape.get(0), vec2.shape.get(0)) && vec1.shape.get(0) == 0) {
            return null;
        }

        Class<?> type = vec1.get(0).getClass();
        Number sum = 0;

        for (int i = 0; i < vec1.shape.get(1); i++) {
            sum = NArray.mulNumsAccumulate(vec1.get(i), vec2.get(i), sum);
        }

        switch (type.getSimpleName()) {
            case "Float" -> {
                return (E) (Float) sum.floatValue();
            }
            case "Double" -> {
                return (E) (Double) sum.doubleValue();
            }
            case "Integer" -> {
                return (E) (Integer) sum.intValue();
            }
            case "Long" -> {
                return (E) (Long) sum.longValue();
            }
            case "Short" -> {
                return (E) (Short) sum.shortValue();
            }
            case "Byte" -> {
                return (E) (Byte) sum.byteValue();
            }
            default -> {
                return null;
            }
        }
    }

    private static <E extends Number> NArray<E> matrixProduct(NArray<E> matrix1, NArray<E> matrix2) {
        if (matrix1 == null || matrix2 == null) {
            return null;
        }

        if (!Objects.equals(matrix1.shape.get(1), matrix2.shape.get(0))) {
            throw new IllegalArgumentException(
                    "Matrices provided do not have a valid size for matrix multiplication" +
                            "(Matrix 1 columns must be equal to Matrix 2 rows)."
            );
        }

        int rows1 = matrix1.array.size();
        int cols1 = ((ArrayList<E>)matrix1.array.get(0)).size();

        int rows2 = matrix2.array.size();
        int cols2 = ((ArrayList<E>)matrix2.array.get(0)).size();

        boolean AB_valid = (cols1 == rows2);
        boolean BA_valid = (cols2 == rows1);

        if (AB_valid) {
            return NArray.multiplyMats(
                    (ArrayList<ArrayList<E>>)matrix1.array,
                    (ArrayList<ArrayList<E>>)matrix2.array,
                    rows1, cols1, rows2, cols2);
        }

        if (BA_valid) {
            return NArray.multiplyMats(
                    (ArrayList<ArrayList<E>>)matrix2.array,
                    (ArrayList<ArrayList<E>>)matrix1.array,
                    rows2, cols2, rows1, cols1);
        }

        throw new IllegalArgumentException(
                "NNMatrix shapes incompatible: (" + rows1 + " x " + cols1 +
                        ") cannot multiply with (" + rows2 + " x " + cols2 + ")"
        );
    }

    private static <E extends Number> NArray<E> multiplyMats(
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
                    sum = NArray.mulNumsAccumulate(matrix1.get(i).get(k), matrix2.get(k).get(j), sum);
                }
                result.get(i).set(j, (E) sum);
            }
        }

        return new NArray<>(result);
    }

    private static <E extends Number> NArray<E> matrixVecProduct(NArray<E> array1, NArray<E> array2) {
        if (array1 == null || array2 == null) {
            return null;
        }

        NArray<E> matrixRef, vecRef;

        if (array1.dimension == 1) {
            vecRef = array1;
            matrixRef = array2;
        } else {
            vecRef = array2;
            matrixRef = array1;
        }

        if (matrixRef.shape.get(1) > 1 && Objects.equals(matrixRef.shape.get(1), vecRef.shape.get(1))) {
            // vec length equals matrix columns
            ArrayList<E> result = new ArrayList<>(matrixRef.shape.get(0));

            for (int i = 0; i < matrixRef.shape.get(0); i++) {
                Number sum = 0;
                for (int j = 0; j < matrixRef.shape.get(1); j++) {
                    sum = NArray.mulNumsAccumulate(matrixRef.get(i, j), vecRef.get(j), sum);
                }
                result.add((E) sum);
            }

            return new NArray<>(result);
        }

        if (matrixRef.shape.get(1) == 1 && Objects.equals(matrixRef.shape.get(1), vecRef.shape.get(0))) {
            // single column matrix * vec
            ArrayList<ArrayList<E>> result = new ArrayList<>(matrixRef.shape.get(0));

            for (int i = 0; i < matrixRef.shape.get(0); i++) {
                ArrayList<E> row = new ArrayList<>(vecRef.shape.get(1));
                for (int j = 0; j < vecRef.shape.get(1); j++) {
                    row.add(NArray.mulNums(matrixRef.get(i, 0), vecRef.get(j)));
                }
                result.add(row);
            }

            return new NArray<>(result);
        }

        throw new IllegalArgumentException(
                "Vector length and matrix columns must match for multiplication of matrix and vector."
        );
    }

    private static <E extends Number> E addNums(E a, E b) {
        Class<?> numType = a.getClass();

        if (numType.equals(Float.class)) {
            return (E) (Float) (a.floatValue() + b.floatValue());
        } else if (numType.equals(Double.class)) {
            return (E) (Double) (a.doubleValue() + b.doubleValue());
        } else if (numType.equals(Integer.class)) {
            return (E) (Integer) (a.intValue() + b.intValue());
        } else if (numType.equals(Long.class)) {
            return (E) (Long) (a.longValue() + b.longValue());
        } else if (numType.equals(Short.class)) {
            return (E) (Short) (short) (a.shortValue() + b.shortValue());
        } else if (numType.equals(Byte.class)) {
            return (E) (Byte) (byte) (a.byteValue() + b.byteValue());
        } else {
            throw new IllegalArgumentException("Cannot add elements of non-numeric data type.");
        }
    }

    private static <E extends Number> E subtractNums(E a, E b) {
        Class<?> numType = a.getClass();

        if (numType.equals(Float.class)) {
            return (E) (Float) (a.floatValue() - b.floatValue());
        } else if (numType.equals(Double.class)) {
            return (E) (Double) (a.doubleValue() - b.doubleValue());
        } else if (numType.equals(Integer.class)) {
            return (E) (Integer) (a.intValue() - b.intValue());
        } else if (numType.equals(Long.class)) {
            return (E) (Long) (a.longValue() - b.longValue());
        } else if (numType.equals(Short.class)) {
            return (E) (Short) (short) (a.shortValue() - b.shortValue());
        } else if (numType.equals(Byte.class)) {
            return (E) (Byte) (byte) (a.byteValue() - b.byteValue());
        } else {
            throw new IllegalArgumentException("Cannot add elements of non-numeric data type.");
        }
    }

    private static <E extends Number> E mulNums(E a, E b) {
        Class<?> numType = a.getClass();

        if (numType.equals(Float.class)) {
            return (E) (Float) (a.floatValue() * b.floatValue());
        } else if (numType.equals(Double.class)) {
            return (E) (Double) (a.doubleValue() * b.doubleValue());
        } else if (numType.equals(Integer.class)) {
            return (E) (Integer) (a.intValue() * b.intValue());
        } else if (numType.equals(Long.class)) {
            return (E) (Long) (a.longValue() * b.longValue());
        } else if (numType.equals(Short.class)) {
            return (E) (Short) (short) (a.shortValue() * b.shortValue());
        } else if (numType.equals(Byte.class)) {
            return (E) (Byte) (byte) (a.byteValue() * b.byteValue());
        } else {
            throw new IllegalArgumentException("Cannot multiply elements of non-numeric data type.");
        }
    }

    private static <E extends Number> Number mulNumsAccumulate(E a, E b, Number accumulator) {
        Class<?> numType = a.getClass();

        if (numType.equals(Float.class)) {
            accumulator = accumulator.floatValue() + (a.floatValue() * b.floatValue());
        } else if (numType.equals(Double.class)) {
            accumulator = accumulator.doubleValue() + (a.doubleValue() * b.doubleValue());
        } else if (numType.equals(Integer.class)) {
            accumulator = accumulator.intValue() + (a.intValue() * b.intValue());
        } else if (numType.equals(Long.class)) {
            accumulator = accumulator.longValue() + (a.longValue() * b.longValue());
        } else if (numType.equals(Short.class)) {
            accumulator = accumulator.shortValue() + (a.shortValue() * b.shortValue());
        } else if (numType.equals(Byte.class)) {
            accumulator = accumulator.byteValue() + (a.byteValue() * b.byteValue());
        } else {
            throw new IllegalArgumentException("Cannot multiply elements of non-numeric data type.");
        }

        return accumulator;
    }
}
