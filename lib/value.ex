defmodule Value do
  @enforce_keys [:data, :ref]

  defstruct [
    :backward,
    :data,
    :operation,
    :ref,
    children: [],
    gradient: 0.0
  ]

  @type t ::
          %__MODULE__{
            backward: fun(any()),
            children: list(%__MODULE__{}),
            data: Integer.t() | float(),
            gradient: non_neg_integer(),
            operation: operation()
          }

  @type operation :: :+ | :* | :- | :/ | :tanh | :pow

  @spec build(integer()) :: __MODULE__.t()
  def build(data) do
    %Value{data: data, ref: make_ref()}
  end

  @spec add(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def add(
        %Value{data: left_data, gradient: left_gradient} = left,
        %Value{data: right_data, gradient: right_gradient} = right
      ) do
    result = %Value{
      children: [left, right],
      data: left_data + right_data,
      operation: :+,
      ref: make_ref()
    }

    backward = fn node ->
      left = %Value{left | gradient: left_gradient + node.gradient}
      right = %Value{right | gradient: right_gradient + node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def add(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec sub(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def sub(
        %Value{data: left_data, gradient: left_gradient} = left,
        %Value{data: right_data, gradient: right_gradient} = right
      ) do
    result = %Value{
      children: [left, right],
      data: left_data - right_data,
      operation: :-,
      ref: make_ref()
    }

    backward = fn node ->
      left = %Value{left | gradient: left_gradient + node.gradient}
      right = %Value{right | gradient: right_gradient + node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def sub(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec mult(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def mult(
        %Value{data: left_data, gradient: left_gradient} = left,
        %Value{data: right_data, gradient: right_gradient} = right
      ) do
    result = %Value{
      children: [left, right],
      data: left_data * right_data,
      operation: :*,
      ref: make_ref()
    }

    backward = fn node ->
      left = %Value{left | gradient: left_gradient + right_data * node.gradient}
      right = %Value{right | gradient: right_gradient + left_data * node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def mult(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec divide(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def divide(
        %Value{data: left_data, gradient: left_gradient} = left,
        %Value{data: right_data, gradient: right_gradient} = right
      ) do
    result = %Value{
      children: [left, right],
      data: trunc(left_data / right_data),
      operation: :/,
      ref: make_ref()
    }

    backward = fn node ->
      left = %Value{left | gradient: left_gradient + right_data * node.gradient}
      right = %Value{right | gradient: right_gradient + left_data * node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def divide(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec tanh(__MODULE__.t()) :: __MODULE__.t()
  def tanh(%Value{data: data, children: children}, gradient \\ 1) do
    t = (:math.exp(2 * data) - 1) / (:math.exp(2 * data) + 1)

    result = %Value{
      data: t,
      gradient: gradient,
      children: children,
      operation: :tanh,
      ref: make_ref()
    }

    backward = fn node ->
      children =
        Enum.map(children, fn child ->
          %Value{child | gradient: 1 - t ** 2}
        end)

      %Value{node | children: children}
    end

    %Value{result | backward: backward}
  end

  def pow(%Value{data: data, gradient: gradient} = child, pow) do
    t = data ** pow

    result = %Value{
      data: t,
      gradient: gradient,
      children: [child],
      operation: :pow,
      ref: make_ref()
    }

    backward = fn node ->
      child = %Value{child | gradient: pow * data ** (pow - 1) * gradient}

      %Value{node | children: [child]}
    end

    %Value{result | backward: backward}
  end

  def backward(%{children: [_left, _right]} = root) do
    root = Map.put(root, :gradient, 1.0)
    %{children: [left, right], gradient: gradient} = root = root.backward.(root)

    %{root | children: [backward(left, gradient), backward(right, gradient)]}
  end

  # def backward(%{children: [_node]} = root) do
  #   root = Map.put(root, :gradient, 1.0)
  #   IO.puts("<<<<<< 1")
  #   %{children: [node], gradient: gradient} = root = root.backward.(root)

  #   %{root | children: [backward(node, gradient)]}
  # end

  def backward(%{backward: nil, gradient: node_gradient} = node, _gradient) do
    %{node | gradient: node_gradient}
  end

  def backward(%{children: [_left, _right]} = root, gradient) do
    %{children: [left, right]} = root = root.backward.(root)

    %{root | children: [backward(left, gradient), backward(right, gradient)]}
  end

  def backward(%{children: [_node]} = root, gradient) do
    %{children: [node]} = root = root.backward.(root)

    %{root | children: [backward(node, gradient)]}
  end

  def sum([head | tail]) do
    Enum.reduce(tail, head, fn value, acc -> add(acc, value) end)
  end

  def map_gradients(%{children: children} = loss) do
    do_flatten(children, %{loss.ref => loss.gradient})
  end

  def do_flatten([] = _loss, acc), do: acc

  def do_flatten(loss, acc) do
    Enum.reduce(loss, acc, fn %{ref: ref, gradient: gradient, children: nodes}, acc ->
      do_flatten(nodes, Map.put(acc, ref, gradient))
    end)
  end
end
