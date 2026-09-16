## Contributing

Contributions are welcome. **Edit [`data/papers.yaml`](data/papers.yaml), not `README.md`** —
the README is generated and your changes to it will be overwritten.

```bash
python scripts/verify.py       # check arXiv ids, titles and the prereq graph
python scripts/gen_readme.py   # regenerate README.md
```

### Guidelines

1. `desc` says **what the paper contributes**, in one line. "A method for X" is not a description.
2. Fill in `prereq`. An entry with no honest prerequisites is rare; one with the wrong ones is worse than none.
3. Give the arXiv id without a version suffix. `verify.py` will confirm the title against the API.
4. Keep the scope: generative video, inference efficiency, post-training.
5. `difficulty` and `minutes` are rough. Err toward honest rather than encouraging.

---

<p align="center">
  <i>Curated for people who read papers to solve a problem.</i>
</p>
