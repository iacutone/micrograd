defmodule MicrogradTest do
  use ExUnit.Case

  @learning_rate -0.1

  test "MLP predicts accurate results after training" do
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
    ]

    # desired targets
    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP.build(%{input_size: 3, layer_sizes: [4, 4, 1]}) |> IO.inspect()

    mlp =
      Enum.reduce(0..1, mlp, fn i, mlp ->
        # forward pass
        predictions =
          Enum.map(xs, fn x ->
            MLP.call(mlp, x)
          end)
          |> List.flatten()

        loss =
          predictions
          |> Enum.zip(ys)
          |> Enum.map(fn {%{data: ypred}, y} ->
            Value.sub(Value.build(ypred), Value.build(y)) |> Value.pow(2)
          end)
          |> Value.sum()

        # |> IO.inspect(label: "<<<< loss")

        # backward pass
        loss = Value.backward(loss)

        IO.inspect("Iteration #{i} loss: #{loss.data}")
        # |> IO.inspect(label: "<<<< backward")
        loss = Value.map_gradients(loss)

        # update weights
        MLP.update(mlp, loss, @learning_rate)
      end)

    # assert_in_delta MLP.call(mlp, [2.0, 3.0, -1.0]).data, 1.0, 0.08
    # assert_in_delta MLP.call(mlp, [3.0, -1.0, 0.5]) |> Value.item(), -1.0, 0.08
    # assert_in_delta MLP.call(mlp, [0.5, 1.0, 1.0]) |> Value.item(), -1.0, 0.08
    # assert_in_delta MLP.call(mlp, [1.0, 1.0, -1.0]) |> Value.item(), 1.0, 0.08
  end
end

# Enum.reduce(1..50, mlp, fn _, mlp ->
#   ypred = Enum.map(xs, &MLP.forward(mlp, &1))

#   loss =
#     Enum.zip(ys, ypred)
#     |> Enum.map(fn {ygt, yout} -> Value.sub(yout, ygt) |> Value.pow(2.0) end)
#     |> Value.sum()

#   loss = Value.backward(loss)

#   MLP.update(mlp, 0.1, loss)
# end)

# maybe keep the parameters, _but_ store the results in an agent as a map?
# use :digraph? https://github.com/princemaple/dg
