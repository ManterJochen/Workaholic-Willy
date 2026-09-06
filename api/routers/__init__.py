"""HTTP surfaces, one module per thing an operator does.

Routers stay thin on purpose: each one translates a request into a library call and the library's answer
into JSON. When a router starts deciding something, such as what counts as blocking or whether a value
is allowed, that decision belongs in ``src/`` instead, where the CLI can reach it too and where it can
be held to account without starting a server.
"""

__all__: list[str] = []
