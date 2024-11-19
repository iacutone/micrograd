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
  def call(%Neuron{weights: weights, bias: %Value{data: bias}}, input) do
    # w * x + b
    data =
      Enum.zip(input, weights)
      |> Enum.map(fn {input, %{data: data}} ->
        input * data
      end)
      |> Enum.sum()
      |> then(fn result -> result + bias end)

    Value.tanh(%Value{data: data, children: weights})
  end

  @spec random_float :: float()
  defp random_float do
    # returns a random float between -1 and 1
    min = -1.0
    max = 1.0
    min + :rand.uniform() * (max - min)
  end
end
