# lattice-uci

A minimal [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface) front
end for [Lattice](https://github.com/hxyulin/lattice): a parser and formatter,
nothing more.

- `parse_command` turns one input line into a `UciCommand`.
- `UciInterface` wraps line-oriented reading and response writing.

It owns no engine state and spawns no threads. The application drives the loop
and decides how to handle each command (the "pull" model), so threading policy
(a worker thread for a long search, an atomic stop flag) stays in the
application.

## Example

A runnable engine loop that answers `go perft`:

```sh
cargo run -p lattice-uci --example basic-engine
```
