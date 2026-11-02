# Contributing

This document outlines the development conventions for `arc-hypothesis-revision` to maintain a clean, collaborative, and reproducible research codebase.

## 1. Development Workflow

Development is organized around GitHub Issues and Pull Requests.

1. **Select an issue** and assign yourself.
2. **Review dependencies** and acceptance criteria.
3. **Create a branch** from the latest `main`.
4. **Implement and test** the required functionality.
5. **Open a pull request** referencing the issue.
6. **Review and merge** once acceptance criteria are satisfied.

Use one branch and one pull request per issue. Avoid committing directly to `main`.

## 2. Branch Naming

Branches follow the convention:

`issue/<number>-<short-description>`

Examples:
- `issue/1-arc-data-schema`
- `issue/2-transformation-library`
- `issue/3-version-space`

Use lowercase letters and hyphens. Branches should originate from the latest `main` unless dependent on unmerged work.

## 3. Commit Conventions

Use descriptive commit messages following Conventional Commits:

`<type>: <short description>`

| Type | Purpose |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fixes |
| `test` | Tests and validation |
| `docs` | Documentation |
| `refactor` | Code improvements without behavior changes |
| `exp` | Research experiments and ablations |

Examples:
- `feat: implement grid validation`
- `test: add transformation consistency tests`
- `fix: correct hypothesis filtering`

Keep commits focused and descriptive.

## 4. Code Conventions

- Follow PEP 8 and use type hints where appropriate.
- Use `snake_case` for functions, variables, and modules; `PascalCase` for classes.
- Keep implementations modular and organized under `src/arc_revision/`.
- Document non-obvious logic and mathematical assumptions.
- Avoid unnecessary dependencies and unrelated changes.

## 5. Testing and Reproducibility

- Add tests for new functionality using `pytest`.
- Validate edge cases and issue acceptance criteria.
- Run relevant tests before submitting a pull request.
- Use explicit random seeds for reproducible experiments.
- Record configurations, parameters, and evaluation settings.
- Maintain consistent experimental conditions when comparing models.
- Avoid committing datasets, checkpoints, or generated artifacts unnecessarily.

## 6. Pull Requests

Use descriptive titles following the commit naming convention.

Each pull request should include:

**Summary:** What was implemented or changed?

**Validation:** What tests were performed?

**Notes:** Any limitations, design decisions, or follow-up work.

Reference the associated issue using `Closes #<number>`.

Pull requests should be reviewed when another contributor is available. Use **Squash and Merge** and delete branches after merging.

## 7. Collaboration

GitHub Issues serve as the source of truth for tasks, dependencies, and progress.

- Assign issues to establish ownership.
- Discuss significant design decisions before implementation.
- Communicate blockers and coordinate overlapping work.
- Keep changes focused, reviewable, and reproducible.