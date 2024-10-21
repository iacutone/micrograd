defmodule LayerTest do
  use ExUnit.Case

  describe "build/1" do
    test "builds a layer of neurons with appropriate weights" do
      layer = Layer.build(%{input_size: 2, output_size: 3})
      assert %Layer{neurons: [%Neuron{}, %Neuron{}, %Neuron{}] = neurons} = layer

      Enum.each(neurons, fn %Neuron{weights: weights} ->
        assert 2 == length(weights)
      end)
    end
  end

  describe "parameters/1" do
    test "returns the weights and biases of all neurons in a given layer" do
      %{neurons: neurons} = layer = Layer.build(%{input_size: 2, output_size: 3})

      neurons =
        Enum.map(neurons, fn %{weights: weights, bias: bias} ->
          weights ++ [bias]
        end)
        |> List.flatten()

      assert ^neurons = Layer.parameters(layer)
    end
  end

  describe "call/2" do
    test "returns the activation value of the layer" do
      layer = Layer.build(%{input_size: 2, output_size: 3})
      result = Layer.call(layer, [1, 1])
      assert 3 = length(result)

      Enum.each(result, fn %{data: activation} ->
        assert_in_delta activation, -1.0, 2
      end)
    end
  end
end
