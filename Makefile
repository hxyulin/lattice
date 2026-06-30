# OpenBench build entrypoint. OpenBench invokes `make EXE=Engine-ABCDEFGH` and
# expects a binary of that name beside this Makefile. This is just a cargo
# wrapper: build the UCI binary, then copy it to $(EXE).
#
# target-cpu=native is set via RUSTFLAGS so it reaches the whole crate - it
# raises worker NPS without changing node counts, since it's the same search.

EXE ?= lattice

openbench:
	RUSTFLAGS="-C target-cpu=native" cargo build --release --bin lattice
	cp target/release/lattice $(EXE)
