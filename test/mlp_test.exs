defmodule MLPTest do
  use ExUnit.Case

  describe "build/1" do
    test "builds an MLP with input size and number of layers" do
      assert %{layers: layers} = MLP.build(%{input_size: 2, layer_sizes: [2, 2, 2]})

      assert 3 = length(layers)

      Enum.each(layers, fn layer ->
        assert %Layer{neurons: [%Neuron{}, %Neuron{}]} = layer
      end)
    end
  end

  describe "parameters/1" do
    test "returns the weights and biases of all layer neurons in a list" do
      %{layers: layers} = mlp = MLP.build(%{input_size: 3, layer_sizes: [4, 4, 1]})

      layers =
        Enum.map(layers, fn layer ->
          Layer.parameters(layer)
        end)
        |> List.flatten()

      assert ^layers = MLP.parameters(mlp)

      assert 41 =
               %{input_size: 3, layer_sizes: [4, 4, 1]}
               |> MLP.build()
               |> MLP.parameters()
               |> length()
    end
  end

  describe "call/2" do
    test "returns the activation values of the layer in the mlp" do
      mlp = MLP.build(%{input_size: 3, layer_sizes: [4, 4, 1]})
      result = MLP.call(mlp, [1, 1, 1])

      assert 3 = length(result)

      result
      |> List.flatten()
      |> Enum.each(fn %{data: activation} ->
        assert_in_delta activation, -1.0, 2
      end)
    end
  end
end
