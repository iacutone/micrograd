defmodule Neuron do
  @enforce_keys [:bias, :weights]

  defstruct [:bias, weights: []]

  @type t :: %__MODULE__{
          bias: Neuron.t(),
          weights: list(Neuron.t())
        }

  @spec build(pos_integer()) :: __MODULE__.t()
  def build(size \\ 1) do
    %__MODULE__{
      weights:
        Enum.reduce(1..size, [], fn _i, acc ->
          [Value.build(random_float()) | acc]
        end),
      bias: Value.build(random_float())
    }
  end

  @spec parameters(__MODULE__.t()) :: list(Value.t())
  def parameters(%Neuron{weights: weights, bias: bias}) do
    weights ++ [bias]
  end

  @spec call(__MODULE__.t(), list()) :: Value.t()
  def call(%Neuron{weights: weights, bias: bias}, input) do
    # w * x + b
    # Handle both raw values and Value structs in input
    input_values = Enum.map(input, fn
      %Value{} = v -> v
      data -> Value.build(data)
    end)

    # Compute weighted sum
    weighted_sum =
      Enum.zip(input_values, weights)
      |> Enum.map(fn {input_val, weight} ->
        Value.mult(input_val, weight)
      end)
      |> Value.sum()

    # Add bias
    linear_output = Value.add(weighted_sum, bias)

    # Apply tanh activation
    Value.tanh(linear_output)
  end

  def update(%__MODULE__{weights: weights, bias: bias} = neuron, loss, learning_rate) do
    %{data: bias_data} = bias

    weights =
      Enum.map(weights, fn %Value{data: data} = weight ->
        gradient = Map.get(loss, weight.ref, 0.0)  # Default to 0 if gradient not found
        Map.put(weight, :data, data - learning_rate * gradient)  # Proper gradient descent
      end)

    bias_gradient = Map.get(loss, bias.ref, 0.0)  # Default to 0 if gradient not found
    bias = Map.put(bias, :data, bias_data - learning_rate * bias_gradient)  # Proper gradient descent

    %{neuron | weights: weights, bias: bias}
  end

  # def update(%__MODULE__{weights: weights, bias: bias} = neuron, learning_rate, gradients) do
  #   weights =
  #     Enum.map(weights, fn %Value{data: data} = weight ->
  #       weight_grad = Map.fetch!(gradients, weight.ref)

  #       %{weight | data: data + -learning_rate * weight_grad}
  #     end)

  #   bias_grad = Map.fetch!(gradients, bias.ref)
  #   bias = %{bias | data: bias.data + -learning_rate * bias_grad}

  #   %__MODULE__{neuron | weights: weights, bias: bias}
  # end

  @spec random_float :: float()
  defp random_float do
    # returns a random float between -1 and 1
    min = -1.0
    max = 1.0
    min + :rand.uniform() * (max - min)
  end
end
