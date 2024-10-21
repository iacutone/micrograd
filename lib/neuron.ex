defmodule Neuron do
  defstruct [:bias, weights: []]

  @type t :: %__MODULE__{
          bias: Neuron.t(),
          weights: list(Neuron.t())
        }

  @spec build(non_neg_integer()) :: __MODULE__.t()
  def build(size \\ 1) do
    %__MODULE__{
      weights:
        Enum.reduce(0..(size - 1), [], fn _i, acc ->
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
  def call(%Neuron{weights: children, bias: %Value{data: bias}}, input) do
    # multi the weights and input then sum the output and the bias
    data =
      Enum.zip(input, children)
      |> Enum.map(fn {input, %{data: data}} ->
        input * data
      end)
      |> Enum.sum()
      |> then(fn result -> result + bias end)

    Value.tanh(%Value{data: data, children: children})
  end

  @spec random_float :: float()
  defp random_float do
    # returns a random float between -1 and 1
    min = -1.0
    max = 1.0
    min + :rand.uniform() * (max - min)
  end
end
