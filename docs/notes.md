# Notes

## 需求（2026-03-18）
压缩并统一项目文档，仅保留 README.md、AGENTS.md、docs/notes.md、docs/logs/2026-03.md。

## 验证
```bash
git status --short -- README.md AGENTS.md docs/notes.md docs/logs/2026-03.md CONTRIBUTING.md CHANGELOG.md docs/workflows.md docs/git_rules.md docs/experiments.md docs/configs.md docs/architecture.md
```

## Git
- branch: `main`
- commit: `git commit -m "docs: shrink and unify project docs"`
