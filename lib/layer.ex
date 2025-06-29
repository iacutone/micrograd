defmodule Layer do
  @enforce_keys [:neurons]

  defstruct [:neurons]

  @type t :: %__MODULE__{
          neurons: list(Neuron.t())
        }

  @spec build(map()) :: list(Layer.t())
  def build(%{input_size: input_size, output_size: output_size}) do
    Enum.reduce(1..output_size, [], fn _i, acc ->
      [Neuron.build(input_size) | acc]
    end)
    |> Enum.reverse()
    |> then(fn neurons -> %Layer{neurons: neurons} end)
  end

  @spec parameters(__MODULE__.t()) :: list()
  def parameters(%__MODULE__{neurons: neurons}) do
    Enum.reduce(neurons, [], fn neuron, acc ->
      [Neuron.parameters(neuron) | acc]
    end)
    |> Enum.reverse()
    |> List.flatten()
  end

  @spec call(__MODULE__.t(), list()) :: list(Value.t()) | Value.t()
  def call(%Layer{neurons: neurons}, input) do
    out =
      Enum.map(neurons, fn neuron ->
        Neuron.call(neuron, input)
      end)

    if length(out) == 1 do
      List.first(out)
    else
      out
    end
  end

  def update(%{neurons: neurons} = layer, loss, learning_rate) do
    IO.inspect(neurons, label: "<<< neurons")
    IO.inspect(loss, label: "<<< loss")

    neurons =
      Enum.map(neurons, fn neuron ->
        Neuron.update(neuron, loss, learning_rate)
      end)

    %{layer | neurons: neurons}
  end
end
