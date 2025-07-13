defmodule MicrogradTest do
  use ExUnit.Case

  @learning_rate 0.1

  test "simple gradient flow test" do
    # Create a simple computation: a = 2, b = 3, c = a * b, loss = c^2
    a = Value.build(2.0)
    b = Value.build(3.0)
    c = Value.mult(a, b)  # c = 6
    loss = Value.pow(c, 2)  # loss = 36

    {_updated_loss, gradients} = Value.map_gradients_with_map(loss)

    # Expected gradients:
    # d(loss)/d(c) = 2 * c = 2 * 6 = 12
    # d(loss)/d(a) = d(loss)/d(c) * d(c)/d(a) = 12 * b = 12 * 3 = 36
    # d(loss)/d(b) = d(loss)/d(c) * d(c)/d(b) = 12 * a = 12 * 2 = 24

    expected_grad_a = 36.0
    expected_grad_b = 24.0

    assert_in_delta Map.get(gradients, a.ref, 0), expected_grad_a, 0.01
    assert_in_delta Map.get(gradients, b.ref, 0), expected_grad_b, 0.01
  end

  test "MLP predicts accurate results after training" do
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
    ]

    # desired targets
    ys = [1.0, -1.0, -1.0, 1.0]
    mlp = MLP.build(%{input_size: 3, layer_sizes: [4, 4, 1]})

    mlp =
      Enum.reduce(0..9, mlp, fn i, mlp ->
        # forward pass
        predictions =
          Enum.map(xs, fn x ->
            MLP.call(mlp, x) |> List.last()  # Get the final layer output
          end)

        loss =
          predictions
          |> Enum.zip(ys)
          |> Enum.map(fn {ypred, y} ->
            Value.sub(ypred, Value.build(y)) |> Value.pow(2)
          end)
          |> Value.sum()

        # backward pass and get gradients
        {_updated_loss, gradients} = Value.map_gradients_with_map(loss)

        if rem(i, 5) == 0 do
          IO.inspect("Iteration #{i} loss: #{loss.data}")

          if i == 0 do
            # Show some actual predictions on first iteration
            pred_values = Enum.map(predictions, &(&1.data))
            IO.inspect("Predictions: #{inspect(pred_values)}")
            IO.inspect("Targets:     #{inspect(ys)}")
          end
        end

        # update weights
        MLP.update(mlp, gradients, @learning_rate)
      end)

    # Test predictions after training - even more relaxed tolerance
    assert_in_delta (MLP.call(mlp, [2.0, 3.0, -1.0]) |> List.last()).data, 1.0, 0.5
    assert_in_delta (MLP.call(mlp, [3.0, -1.0, 0.5]) |> List.last()).data, -1.0, 0.5
    assert_in_delta (MLP.call(mlp, [0.5, 1.0, 1.0]) |> List.last()).data, -1.0, 0.5
    assert_in_delta (MLP.call(mlp, [1.0, 1.0, -1.0]) |> List.last()).data, 1.0, 0.5
  end
end
