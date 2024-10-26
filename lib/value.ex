defmodule Value do
  @enforce_keys [:data]

  defstruct [:backward, :data, :operation, children: [], gradient: 0]

  @type t ::
          %__MODULE__{
            backward: fun(any()),
            children: list(%__MODULE__{}),
            data: Integer.t() | float(),
            gradient: non_neg_integer(),
            operation: operation()
          }

  @type operation :: :+ | :* | :- | :/ | :tanh

  @spec build(integer()) :: __MODULE__.t()
  def build(data) do
    %Value{data: data}
  end

  @spec add(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def add(%Value{data: left_data} = left, %Value{data: right_data} = right) do
    result = %Value{
      children: [left, right],
      data: left_data + right_data,
      operation: :+
    }

    backward = fn node ->
      left = %Value{left | gradient: right.gradient + node.gradient}
      right = %Value{right | gradient: right.gradient + node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def add(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec sub(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def sub(%Value{data: left_data} = left, %Value{data: right_data} = right) do
    result = %Value{
      children: [left, right],
      data: left_data - right_data,
      operation: :-
    }

    backward = fn node ->
      left = %Value{left | gradient: right.gradient + node.gradient}
      right = %Value{right | gradient: right.gradient + node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def sub(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec mult(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def mult(%Value{data: left_data} = left, %Value{data: right_data} = right) do
    result = %Value{
      children: [left, right],
      data: left_data * right_data,
      operation: :*
    }

    backward = fn node ->
      left = %Value{left | gradient: right.data * node.gradient}
      right = %Value{right | gradient: left.data * node.gradient}

      %Value{node | children: [left, right]}
    end

    %Value{result | backward: backward}
  end

  def mult(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec divide(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def divide(%Value{data: left_data} = left, %Value{data: right_data} = right) do
    result = %Value{
      children: [left, right],
      data: trunc(left_data / right_data),
      operation: :/
    }

    backward = fn node ->
      left = %Value{left | gradient: right.data * node.gradient}
      right = %Value{right | gradient: left.data * node.gradient}

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
      operation: :tanh
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
end
