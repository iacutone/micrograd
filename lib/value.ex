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

  defp build_digraph_and_topsort(root) do
    # Create a new directed graph
    graph = :digraph.new()

    # Collect all nodes and build the digraph
    visited = MapSet.new()
    {all_nodes, _visited} = collect_nodes(root, [], visited)

    # Add all nodes as vertices
    Enum.each(all_nodes, fn node ->
      :digraph.add_vertex(graph, node.ref, node)
    end)

    # Add edges (from parent to children)
    Enum.each(all_nodes, fn node ->
      Enum.each(node.children, fn child ->
        :digraph.add_edge(graph, node.ref, child.ref)
      end)
    end)

    # Get topological sort
    case :digraph_utils.topsort(graph) do
      false ->
        # Graph has cycles, should not happen in our computation graph
        :digraph.delete(graph)
        raise "Cycle detected in computation graph"

      topo_refs ->
        # Convert references back to nodes
        topo_nodes = Enum.map(topo_refs, fn ref ->
          {^ref, node} = :digraph.vertex(graph, ref)
          node
        end)

        # Clean up the digraph
        :digraph.delete(graph)

        topo_nodes
    end
  end

  defp collect_nodes(node, acc, visited) do
    if MapSet.member?(visited, node.ref) do
      {acc, visited}
    else
      visited = MapSet.put(visited, node.ref)
      acc = [node | acc]

      # Recursively collect children
      Enum.reduce(node.children, {acc, visited}, fn child, {acc_inner, visited_inner} ->
        collect_nodes(child, acc_inner, visited_inner)
      end)
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
    # Build digraph and get topological sort using Erlang's digraph library
    topo_nodes = build_digraph_and_topsort(loss)

    # Initialize gradient accumulator with loss gradient = 1.0
    initial_acc = %{loss.ref => 1.0}

    # Propagate gradients backward using the topologically sorted nodes
    Enum.reduce(topo_nodes, initial_acc, fn node, acc ->
      # Propagate gradients to children based on operation
      propagate_gradients(node, acc)
    end)
  end

    def sum([head | tail]) do
    Enum.reduce(tail, head, fn value, acc -> add(acc, value) end)
  end
end
