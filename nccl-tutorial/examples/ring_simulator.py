#!/usr/bin/env python3
import argparse
import random


def ring_allreduce(inputs, verbose=False):
    ranks = len(inputs)
    if ranks == 0 or not inputs[0]:
        raise ValueError("inputs must contain nonempty rank arrays")
    count = len(inputs[0])
    if any(len(row) != count for row in inputs) or count % ranks:
        raise ValueError("equal input lengths divisible by the rank count are required")
    width = count // ranks
    chunks = [[row[c * width:(c + 1) * width] for c in range(ranks)] for row in inputs]
    sources = [[{r} for _ in range(ranks)] for r in range(ranks)]
    expected = [sum(row[i] for row in inputs) for i in range(count)]

    if verbose:
        print(f"P={ranks}, elements per chunk={width}")
        for r, row in enumerate(inputs):
            print(f"input r{r}: {row}")
        print("\nReduceScatter (snapshot all sends before receiving):")
    for step in range(ranks - 1):
        messages = []
        for src in range(ranks):
            dst = (src + 1) % ranks
            chunk = (src - step) % ranks
            messages.append((src, dst, chunk, chunks[src][chunk][:], sources[src][chunk].copy()))
        if verbose:
            print(f"  step {step}: " + "; ".join(f"r{s}->r{d}: c{c}" for s, d, c, _, _ in messages))
        for _, dst, chunk, values, contributors in messages:
            assert sources[dst][chunk].isdisjoint(contributors)
            chunks[dst][chunk] = [a + b for a, b in zip(chunks[dst][chunk], values)]
            sources[dst][chunk].update(contributors)
    for r in range(ranks):
        owned = (r + 1) % ranks
        assert sources[r][owned] == set(range(ranks))
        assert chunks[r][owned] == expected[owned * width:(owned + 1) * width]
        if verbose:
            print(f"  r{r} owns reduced c{owned}: {chunks[r][owned]}")

    if verbose:
        print("\nAllGather:")
    for step in range(ranks - 1):
        messages = []
        for src in range(ranks):
            dst = (src + 1) % ranks
            chunk = (src + 1 - step) % ranks
            messages.append((src, dst, chunk, chunks[src][chunk][:]))
        if verbose:
            print(f"  step {step}: " + "; ".join(f"r{s}->r{d}: c{c}" for s, d, c, _ in messages))
        for _, dst, chunk, values in messages:
            chunks[dst][chunk] = values
    outputs = [[value for chunk in row for value in chunk] for row in chunks]
    assert all(row == expected for row in outputs)
    if verbose:
        for r, row in enumerate(outputs):
            print(f"output r{r}: {row}")
        print(f"PASS: {2 * (ranks - 1)} rounds; sent per rank = {2 * (ranks - 1) * width} elements")
    return outputs


def self_test():
    rng = random.Random(20260922)
    cases = 0
    for ranks in range(1, 10):
        for width in (1, 2, 7):
            for _ in range(3):
                inputs = [[rng.randrange(-100, 101) for _ in range(ranks * width)] for _ in range(ranks)]
                expected = [sum(values) for values in zip(*inputs)]
                assert ring_allreduce(inputs) == [expected] * ranks
                cases += 1
    for invalid in ([], [[]], [[1], [2, 3]], [[1], [2]]):
        try:
            ring_allreduce(invalid)
        except ValueError:
            cases += 1
        else:
            raise AssertionError("invalid input was accepted")
    print(f"PASS: {cases} cases (P=1..9, multiple chunk sizes, signed data, invalid shapes)")


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description="Integer Ring AllReduce teaching model, not a performance simulator")
    parser.add_argument("--ranks", type=positive_int, default=4)
    parser.add_argument("--chunk-size", type=positive_int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        inputs = [[10 * r + i for i in range(args.ranks * args.chunk_size)] for r in range(args.ranks)]
        ring_allreduce(inputs, verbose=True)


if __name__ == "__main__":
    main()
