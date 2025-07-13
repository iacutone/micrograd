defmodule MLP do
  @enforce_keys [:layers]

  defstruct [:layers]

  @type t :: %__MODULE__{layers: list(Layer.t())}

  @spec build(%{input_size: pos_integer(), layer_sizes: list()}) :: list(Layer.t())
  def build(%{input_size: input_size, layer_sizes: layer_sizes}) do
    layers =
      [input_size | layer_sizes]
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.reduce([], fn [i_size, output_size], acc ->
        [Layer.build(%{input_size: i_size, output_size: output_size}) | acc]
      end)
      |> Enum.reverse()

    %MLP{layers: layers}
  end

  @spec parameters(__MODULE__.t()) :: list()
  def parameters(%{layers: layers}) do
    Enum.reduce(layers, [], fn layer, acc ->
      [Layer.parameters(layer) | acc]
    end)
    |> Enum.reverse()
    |> List.flatten()
  end

  @spec call(__MODULE__.t(), list()) :: list()
  def call(%__MODULE__{layers: layers}, inputs) do
    # Convert initial inputs to Value structs if they aren't already
    current_values = Enum.map(inputs, fn
      %Value{} = v -> v
      data -> Value.build(data)
    end)

    {_, all_outputs} = Enum.reduce(layers, {current_values, []}, fn layer, {current_inputs, outputs} ->
      output = Layer.call(layer, current_inputs)
      # Convert single Value to list for next layer, or keep as list
      next_inputs = if is_struct(output, Value) do
        [output]
      else
        output
      end
      {next_inputs, [output | outputs]}
    end)

    all_outputs |> Enum.reverse()
  end

  def update(%{layers: layers} = mlp, gradients, learning_rate) do
    layers =
      Enum.map(layers, fn layer ->
        Layer.update(layer, gradients, learning_rate)
      end)

    %{mlp | layers: layers}
  end
end
