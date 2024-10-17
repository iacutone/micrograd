defmodule MicrogradTest do
  use ExUnit.Case
  doctest Micrograd

  test "greets the world" do
    assert Micrograd.hello() == :world
  end
end
