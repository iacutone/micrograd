defmodule NeuronTest do
  use ExUnit.Case

  describe "build/1" do
    test "builds a Neuron with appropriately sized weights and a bias" do
      assert %Neuron{
               weights: [%Value{data: weight_1}, %Value{data: weight_2}],
               bias: %Value{data: bias}
             } = Neuron.build(2)

      assert_in_delta weight_1, weight_2, 2
      assert_in_delta bias, 1.0, 2
    end
  end

  describe "parameters/1" do
    test "appends the bias to the end of the weights list" do
      assert %Neuron{
               weights: [%Value{data: weight_1}, %Value{data: weight_2}],
               bias: %Value{data: bias}
             } = neuron = Neuron.build(2)

      assert [%Value{data: ^weight_1}, %Value{data: ^weight_2}, %Value{data: ^bias}] =
               Neuron.parameters(neuron)

      assert 5 = 4 |> Neuron.build() |> Neuron.parameters() |> length()
    end
  end

  describe "call/1" do
    test "returns the tanh Value struct with the input Neuron as children values" do
      %{data: activation} = Neuron.build(2) |> Neuron.call([1, 1])
      assert_in_delta activation, -1.0, 2
    end
  end
end
