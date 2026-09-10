package org.sdws.mathematics;

import java.util.Objects;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Collection;

public class NNVector<E> {
    private ArrayList<E> vector;
    public ArrayList<Integer> shape;

    private static final Random rng = new Random();

    private NNVector(ArrayList<E> vector) {
        this.vector = vector;
        this.shape = new ArrayList<>();
        this.shape.add(1);
        this.shape.add(this.vector.size());
    }

    /*
    Class methods
     */

    @Override
    public String toString() {
        return this.vector.toString();
    }

    public void set(int index, E value) {
        if (index < 0 || index >= this.vector.size()) {
            throw new IndexOutOfBoundsException(
                    "Element index: " + index + " out of bounds of the vector."
            );
        }

        this.vector.set(index, value);
    }

    public E get(int index) {
        if (index < 0 || index >= this.vector.size()) {
            throw new IndexOutOfBoundsException(
                    "Element index: " + index + " out of bounds of the vector."
            );
        }

        return this.vector.get(index);
    }

    /**
     * Returns the number of elements contained in this vector object.
     * @return a single {@link Integer} value
     */
    public Integer size() {
        return this.vector.size();
    }

    /*
    Static class functions
     */

    /**
     *
     * @param elements any amount of objects of the provided type parameter,
     * in the form of a {@code varargs} argument.<br></br>
     * Essentially, the amount of arguments passed corresponds to the amount of elements
     * the final {@link NNVector} object will have.
     * @return a new {@link NNVector} object.
     * @param <E> the type of the elements of this vector.
     */
    @SafeVarargs
    public static <E> NNVector<E> FromElements(E... elements) {
        if (elements.length == 0) {
            return new NNVector<>(new ArrayList<>());
        }

        return new NNVector<>(new ArrayList<>(List.of(elements)));
    }

    public static <E> NNVector<E> FromCollection(Collection<E> elements) {
        if (elements.isEmpty()) {
            return new NNVector<>(new ArrayList<>());
        }

        return new NNVector<>(new ArrayList<>(elements));
    }

    public static NNVector<Float> CreateZeros(int numElements) {
        ArrayList<Float> vector = new ArrayList<>(numElements);
        for (int i = 0; i < numElements; i++) {
            vector.add(0f);
        }

        return new NNVector<>(vector);
    }

    public static NNVector<Float> CreateRandom(int numElements) {
        ArrayList<Float> vector = new ArrayList<>(numElements);
        for (int i = 0; i < numElements; i++) {
            vector.add((rng.nextFloat() * 2) - 1);
        }

        return new NNVector<>(vector);
    }

    public static <E> NNVector<E> DeepCopy(NNVector<E> vector) {
        if (vector == null) {
            return null;
        }

        ArrayList<E> copy = new ArrayList<>(vector.vector.size());
        for (E element : vector.vector) {
            copy.add(element);
        }

        return new NNVector<>(copy);
    }

    public static <E> NNVector<E> ConvertToVector(NNMatrix<E> matrix) {
        if (matrix.shape.get(0) > 1) {
            throw new IllegalArgumentException(
                    "Cannot convert to vector (shape is currently " + matrix.shape.get(0)+" x "+ matrix.shape.get(1) + ").");
        }

        return NNVector.FromCollection(matrix.get(0));
    }

    public static <E extends Number> NNVector<E> Add(NNVector<E> vector1, NNVector<E> vector2) {
        if (vector1 == null || vector2 == null) {
            return null;
        }

        if (!Objects.equals(vector1.vector.size(), vector2.vector.size())) {
            throw new IllegalArgumentException("Vectors are not of equal size for addition.");
        }

        if (vector1.vector.isEmpty() || vector2.vector.isEmpty()) {
            return NNVector.FromElements();
        }

        ArrayList<E> result = new ArrayList<>(vector1.vector.size());
        Class<?> type = vector1.get(0).getClass();

        for (int i = 0; i < vector1.vector.size(); i++) {
            if (type.equals(Float.class)) {
                result.add((E) (Float) (vector1.get(i).floatValue() + vector2.get(i).floatValue()));
            } else if (type.equals(Double.class)) {
                result.add((E) (Double) (vector1.get(i).doubleValue() + vector2.get(i).doubleValue()));
            } else if (type.equals(Integer.class)) {
                result.add((E) (Integer) (vector1.get(i).intValue() + vector2.get(i).intValue()));
            } else if (type.equals(Long.class)) {
                result.add((E) (Long) (vector1.get(i).longValue() + vector2.get(i).longValue()));
            } else if (type.equals(Short.class)) {
                result.add((E) (Short) (short) (vector1.get(i).shortValue() + vector2.get(i).shortValue()));
            } else if (type.equals(Byte.class)) {
                result.add((E) (Byte) (byte) (vector1.get(i).byteValue() + vector2.get(i).byteValue()));
            } else {
                throw new IllegalArgumentException("Cannot add vector elements of non-numeric data type.");
            }
        }

        return new NNVector<>(result);
    }

    public static <E extends Number> E Dot(NNVector<E> vector1, NNVector<E> vector2) {
        if (vector1 == null || vector2 == null) {
            return null;
        }

        if (!Objects.equals(vector1.vector.size(), vector2.vector.size())) {
            throw new IllegalArgumentException("Vectors are not of equal size for dot product.");
        }

        if (vector1.vector.isEmpty() || vector2.vector.isEmpty()) {
            return null;
        }

        Class<?> type = vector1.get(0).getClass();
        Number sum = 0;

        for (int i = 0; i < vector1.vector.size(); i++) {
            if (type.equals(Float.class)) {
                sum = sum.floatValue() + (vector1.get(i).floatValue() * vector2.get(i).floatValue());
            } else if (type.equals(Double.class)) {
                sum = sum.doubleValue() + (vector1.get(i).doubleValue() * vector2.get(i).doubleValue());
            } else if (type.equals(Integer.class)) {
                sum = sum.intValue() + (vector1.get(i).intValue() * vector2.get(i).intValue());
            } else if (type.equals(Long.class)) {
                sum = sum.longValue() + (vector1.get(i).longValue() * vector2.get(i).longValue());
            } else if (type.equals(Short.class)) {
                sum = sum.shortValue() + (vector1.get(i).shortValue() * vector2.get(i).shortValue());
            } else if (type.equals(Byte.class)) {
                sum = sum.byteValue() + (vector1.get(i).byteValue() * vector2.get(i).byteValue());
            } else {
                throw new IllegalArgumentException("Cannot multiply vector elements of non-numeric data type.");
            }
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
}
