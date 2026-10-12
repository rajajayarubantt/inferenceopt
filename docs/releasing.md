# Releasing

Releases are built by GitHub Actions and published with
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/): no API tokens are
stored anywhere.

## One-time setup

1. **GitHub environments** (repo Settings > Environments): create `testpypi` and `pypi`.
   Give `pypi` required reviewers so every production release needs approval.
2. **TestPyPI**: at <https://test.pypi.org/manage/account/publishing/>, add a pending
   publisher: project `inferenceopt`, owner `rajajayarubantt`, repository `inferenceopt`,
   workflow `release.yml`, environment `testpypi`.
3. **PyPI**: same at <https://pypi.org/manage/account/publishing/> with environment
   `pypi`.
4. When ready to publish to the real PyPI (planned for 0.1.0), set the repository
   variable `PUBLISH_TO_PYPI` to `true`.

## Cutting a release

1. Move the `Unreleased` entries in `CHANGELOG.md` under the new version and date.
2. Tag and push:

    ```bash
    git tag -s v0.0.1 -m "v0.0.1"
    git push origin v0.0.1
    ```

3. The `Release` workflow builds the UI and distributions, smoke-tests the wheel on
   Linux, Windows and macOS, publishes to TestPyPI, (if enabled, after approval) to
   PyPI, and creates a draft GitHub release. Review and publish the draft.

The version comes from the git tag via `hatch-vcs`; never edit it by hand.
