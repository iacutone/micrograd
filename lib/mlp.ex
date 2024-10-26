defmodule MLP do
  @enforce_keys [:layers]

  defstruct [:layers]

  @type t :: %__MODULE__{layers: list(Layer.t())}

  @spec build(%{input_size: non_neg_integer(), layer_sizes: list()}) :: list(Layer.t())
  def build(%{input_size: input_size, layer_sizes: layer_sizes}) do
    layers =
      Enum.reduce(layer_sizes, [], fn layer_size, acc ->
        [Layer.build(%{input_size: input_size, output_size: layer_size}) | acc]
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

  @spec call(__MODULE__.t(), list()) :: list(Value.t())
  def call(%__MODULE__{layers: layers}, inputs) do
    Enum.map(layers, fn layer ->
      Layer.call(layer, inputs)
    end)
  end
end
