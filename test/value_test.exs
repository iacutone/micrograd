defmodule ValueTest do
  use ExUnit.Case

  describe "add/2" do
    test "adds the Value struct input data values and returns a new Value struct with summed :data value" do
      a = Value.build(1)
      b = Value.build(1)
      assert 2 = Value.add(a, b).data
    end

    test "raises an error when not passed Value structs" do
      assert_raise RuntimeError, "must pass Value structs arguments", fn ->
        Value.add(1, 2)
      end
    end

    test "sets the :operation as :+" do
      a = Value.build(1)
      b = Value.build(1)
      assert :+ = Value.add(a, b).operation
    end

    test "sets the left and right as :children list elements" do
      a = Value.build(1)
      b = Value.build(1)
      assert [^a, ^b] = Value.add(a, b).children
    end

    test "when backward is called sets the :children's gradient value" do
      a = Value.build(1)
      b = Value.build(1)
      result = Value.add(a, b)
      result = Map.put(result, :gradient, 1)
      result = result.backward.(result)

      assert %{children: [%{gradient: 1}, %{gradient: 1}]} = result
    end
  end

  describe "sub/2" do
    test "adds the Value struct input data values and returns a new Value struct with summed :data value" do
      a = Value.build(1)
      b = Value.build(1)
      assert 0 = Value.sub(a, b).data
    end

    test "raises an error when not passed Value structs" do
      assert_raise RuntimeError, "must pass Value structs arguments", fn ->
        Value.sub(1, 2)
      end
    end

    test "sets the :operation as :-" do
      a = Value.build(1)
      b = Value.build(1)
      assert :- = Value.sub(a, b).operation
    end

    test "sets the left and right as :children list elements" do
      a = Value.build(1)
      b = Value.build(1)
      assert [^a, ^b] = Value.sub(a, b).children
    end

    test "when backward is called sets the :children's gradient value" do
      a = Value.build(1)
      b = Value.build(1)
      result = Value.add(a, b)
      result = Map.put(result, :gradient, 1)
      result = result.backward.(result)

      assert %{children: [%{gradient: 1}, %{gradient: 1}]} = result
    end
  end

  describe "mult/2" do
    test "adds the Value struct input data values and returns a new Value struct with summed :data value" do
      a = Value.build(2)
      b = Value.build(2)
      assert 4 = Value.mult(a, b).data
    end

    test "raises an error when not passed Value structs" do
      assert_raise RuntimeError, "must pass Value structs arguments", fn ->
        Value.mult(1, 2)
      end
    end

    test "sets the :operation as :*" do
      a = Value.build(1)
      b = Value.build(1)
      assert :* = Value.mult(a, b).operation
    end

    test "sets the left and right as :children list elements" do
      a = Value.build(1)
      b = Value.build(1)
      assert [^a, ^b] = Value.mult(a, b).children
    end

    test "when backward is called sets the :children's gradient value" do
      a = Value.build(1)
      b = Value.build(2)
      result = Value.mult(a, b)
      result = Map.put(result, :gradient, 2)
      result = result.backward.(result)

      assert %{children: [%{gradient: 4}, %{gradient: 2}]} = result
    end
  end

  describe "divide/2" do
    test "adds the Value struct input data values and returns a new Value struct with summed :data value" do
      a = Value.build(2)
      b = Value.build(2)
      assert 1 = Value.divide(a, b).data
    end

    test "raises an error when not passed Value structs" do
      assert_raise RuntimeError, "must pass Value structs arguments", fn ->
        Value.mult(1, 2)
      end
    end

    test "sets the :operation as :/" do
      a = Value.build(1)
      b = Value.build(1)
      assert :/ = Value.divide(a, b).operation
    end

    test "sets the left and right as :children list elements" do
      a = Value.build(1)
      b = Value.build(1)
      assert [^a, ^b] = Value.divide(a, b).children
    end

    test "when backward is called sets the :children's gradient value" do
      a = Value.build(1)
      b = Value.build(2)
      result = Value.divide(a, b)
      result = Map.put(result, :gradient, 2)
      result = result.backward.(result)

      assert %{children: [%{gradient: 4}, %{gradient: 2}]} = result
    end
  end

  describe "tanh/1" do
    test "returns tanh of the prior node's data value" do
      a = Value.build(1)
      b = Value.build(1)
      c = Value.add(a, b)
      assert 0.9640275800758169 == Value.tanh(c).data
    end

    test "sets the :operation as :tanh" do
      a = Value.build(1)
      b = Value.build(1)
      c = Value.add(a, b)
      assert :tanh = Value.tanh(c).operation
    end

    test "when backward is called sets the :children's gradient value" do
      a = Value.build(1)
      b = Value.build(1)
      c = Value.add(a, b)
      result = Value.tanh(c)
      result = result.backward.(result)

      assert %{children: [%{gradient: 0.07065082485316443}, %{gradient: 0.07065082485316443}]} =
               result
    end
  end
end
