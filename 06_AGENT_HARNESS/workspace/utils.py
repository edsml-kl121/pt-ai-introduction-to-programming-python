def greet(name: str | None) -> str:
    if name is None:
        return "Hello, there!"
    return f"Hello, {name}!"
