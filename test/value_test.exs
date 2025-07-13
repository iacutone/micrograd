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
      result = Map.put(result, :gradient, 1.0)
      result = result.backward.(result)

      assert %{children: [%{gradient: 1.0}, %{gradient: 1.0}]} = result
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
      result = Value.sub(a, b)
      result = Map.put(result, :gradient, 1.0)
      result = result.backward.(result)

      assert %{children: [%{gradient: 1.0}, %{gradient: -1.0}]} = result
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

      assert %{children: [%{gradient: 4.0}, %{gradient: 2.0}]} = result
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

      assert %{children: [%{gradient: 1.0}, %{gradient: -0.5}]} = result
    end
  end

  describe "tanh/2" do
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
      result = Map.put(result, :gradient, 1.0)
      result = result.backward.(result)

      assert %{children: [%{gradient: 0.07065082485316443}]} = result
    end
  end

  describe "pow/2" do
    setup do
      a = %Value{data: 2, gradient: 0.0, ref: make_ref()}
      result = Value.pow(a, 2)

      %{a: a, result: result}
    end

    test "sets data correctly", %{result: result} do
      assert %{data: 4} = result
    end

    test "sets operation correctly", %{result: result} do
      assert %{operation: :pow} = result
    end

    test "sets the children to the pow/2 value", %{a: a, result: result} do
      assert %{children: [^a]} = result
    end

    test "sets gradient value correctly", %{result: result} do
      assert %{gradient: 0.0} = result
      result = Map.put(result, :gradient, 12.0)
      result = result.backward.(result)
      assert %{data: 4, gradient: 12.0, children: [%{data: 2, gradient: 48.0}]} = result
    end
  end

  describe "backward/1" do
    test "sets the gradient to 1.0 for the root node" do
      a = Value.build(1.0)
      assert %{gradient: 1.0} = Value.build(1.0) |> Value.add(a) |> Value.backward()
    end

    test "gradients accumulate correctly" do
      a = Value.build(1.0)
      b = Value.add(a, a)

      assert %{data: 2.0, gradient: 1.0, children: [%{gradient: 2.0}, %{gradient: 2.0}]} =
               Value.backward(b)

      a = Value.build(2.0)
      b = Value.build(-3.0)
      c = Value.build(10.0)

      assert %Value{
               children: [
                 %Value{
                   children: [
                     %Value{
                       backward: nil,
                       children: [],
                       data: 2.0,
                       gradient: -3.0,
                       operation: nil
                     },
                     %Value{backward: nil, children: [], data: -3.0, gradient: 2.0}
                   ],
                   data: -6.0,
                   gradient: 1.0,
                   operation: :*
                 },
                 %Value{backward: nil, children: [], data: 10.0, gradient: 1.0, operation: nil}
               ],
               data: 4.0,
               gradient: 1.0,
               operation: :+
             } = Value.mult(a, b) |> Value.add(c) |> Value.backward()
    end
  end

  describe "sum/1" do
    test "sums the value nodes correctly" do
      a = Value.build(2)
      b = Value.build(2)

      assert %{data: 4, children: [^a, ^b]} = Value.sum([a, b])
    end
  end
end
