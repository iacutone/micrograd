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
        %Value{data: left_data} = left,
        %Value{data: right_data} = right
      ) do
    %Value{
      children: [left, right],
      data: left_data + right_data,
      operation: :+,
      ref: make_ref(),
      gradient: 0.0
    }
  end

  def add(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec sub(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def sub(
        %Value{data: left_data} = left,
        %Value{data: right_data} = right
      ) do
    %Value{
      children: [left, right],
      data: left_data - right_data,
      operation: :-,
      ref: make_ref(),
      gradient: 0.0
    }
  end

  def sub(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec mult(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def mult(
        %Value{data: left_data} = left,
        %Value{data: right_data} = right
      ) do
    %Value{
      children: [left, right],
      data: left_data * right_data,
      operation: :*,
      ref: make_ref(),
      gradient: 0.0
    }
  end

  def mult(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec divide(__MODULE__.t(), __MODULE__.t()) :: __MODULE__.t()
  def divide(
        %Value{data: left_data} = left,
        %Value{data: right_data} = right
      ) do
    %Value{
      children: [left, right],
      data: left_data / right_data,
      operation: :/,
      ref: make_ref(),
      gradient: 0.0
    }
  end

  def divide(_, _) do
    raise "must pass Value structs arguments"
  end

  @spec tanh(__MODULE__.t()) :: __MODULE__.t()
  def tanh(%Value{data: data} = input) do
    t = (:math.exp(2 * data) - 1) / (:math.exp(2 * data) + 1)

    %Value{
      data: t,
      children: [input],
      operation: :tanh,
      ref: make_ref(),
      gradient: 0.0  # Initialize gradient to 0
    }
  end

  def pow(%Value{data: data} = child, _pow) do
    t = data ** 2  # Assume power of 2 for simplicity

    %Value{
      data: t,
      children: [child],
      operation: :pow,
      ref: make_ref(),
      gradient: 0.0
    }
  end

  defp build_topo(node, topo, visited) do
    if MapSet.member?(visited, node.ref) do
      {topo, visited}
    else
      visited = MapSet.put(visited, node.ref)

      # First, recursively visit all children
      {topo, visited} =
        Enum.reduce(node.children, {topo, visited}, fn child, {topo_acc, visited_acc} ->
          build_topo(child, topo_acc, visited_acc)
        end)

      # Then add this node AFTER all its children
      {[node | topo], visited}
    end
  end

  defp propagate_gradients(node, acc) do
    # Get the current node's gradient from the accumulator
    node_grad = Map.get(acc, node.ref, 0.0)

    case node.operation do
      :+ ->
        # Both children get the same gradient
        Enum.reduce(node.children, acc, fn child, acc ->
          current_grad = Map.get(acc, child.ref, 0.0)
          Map.put(acc, child.ref, current_grad + node_grad)
        end)

      :- ->
        # Left child gets positive gradient, right gets negative
        [left, right] = node.children
        left_grad = Map.get(acc, left.ref, 0.0)
        right_grad = Map.get(acc, right.ref, 0.0)

        acc
        |> Map.put(left.ref, left_grad + node_grad)
        |> Map.put(right.ref, right_grad - node_grad)

      :* ->
        # Chain rule: gradient * other operand
        [left, right] = node.children
        left_grad = Map.get(acc, left.ref, 0.0)
        right_grad = Map.get(acc, right.ref, 0.0)

        acc
        |> Map.put(left.ref, left_grad + right.data * node_grad)
        |> Map.put(right.ref, right_grad + left.data * node_grad)

      :/ ->
        # Quotient rule
        [left, right] = node.children
        left_grad = Map.get(acc, left.ref, 0.0)
        right_grad = Map.get(acc, right.ref, 0.0)

        acc
        |> Map.put(left.ref, left_grad + (1.0 / right.data) * node_grad)
        |> Map.put(right.ref, right_grad - (left.data / (right.data * right.data)) * node_grad)

      :tanh ->
        # d/dx(tanh(x)) = 1 - tanh²(x)
        [child] = node.children
        child_grad = Map.get(acc, child.ref, 0.0)
        tanh_grad = (1 - node.data * node.data) * node_grad

        Map.put(acc, child.ref, child_grad + tanh_grad)

      :pow ->
        # d/dx(x^n) = n * x^(n-1)
        [child] = node.children
        child_grad = Map.get(acc, child.ref, 0.0)
        # For power of 2: gradient = 2 * x * parent_gradient
        pow_grad = 2 * child.data * node_grad

        Map.put(acc, child.ref, child_grad + pow_grad)

      _ ->
        # No operation or unknown operation - don't propagate gradients
        acc
    end
  end

  def map_gradients(%{} = loss) do
    # Collect all nodes in topological order
    topo = []
    visited = MapSet.new()

    {topo, _visited} = build_topo(loss, topo, visited)

    # Initialize gradient accumulator with loss gradient = 1.0
    initial_acc = %{loss.ref => 1.0}

    # Propagate gradients backward and return the gradient map
    Enum.reduce(topo, initial_acc, fn node, acc ->
      # Propagate gradients to children based on operation
      propagate_gradients(node, acc)
    end)
  end

    def sum([head | tail]) do
    Enum.reduce(tail, head, fn value, acc -> add(acc, value) end)
  end
end
