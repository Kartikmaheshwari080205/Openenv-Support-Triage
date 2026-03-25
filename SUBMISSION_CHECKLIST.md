# Submission Checklist

Use this checklist as the final gate before submitting your OpenEnv environment.

## 1) Local precheck (required)

- [ ] Run PowerShell precheck:

```bash
./precheck.ps1
```

- [ ] Confirm output includes:
  - `[OK] openenv-support-triage: Ready for multi-mode deployment`
  - `[5/5] Precheck passed`

- [ ] Verify baseline JSON prints 3 task scores and average score in `[0.0, 1.0]`.

## 2) Linux/macOS precheck (optional but recommended)

- [ ] Run shell precheck:

```bash
chmod +x precheck.sh
./precheck.sh
```

- [ ] Confirm validator/build/endpoints/baseline pass.

## 3) CI check (recommended)

- [ ] Push repo to GitHub.
- [ ] Confirm workflow passes:
  - `.github/workflows/precheck.yml`

## 4) Hugging Face Space deployment (required)

- [ ] Create Docker Space and push this repo.
- [ ] Confirm README frontmatter includes `sdk: docker` and `app_port: 7860`.
- [ ] Add secret in Space settings:
  - `OPENAI_API_KEY`
- [ ] Wait for Space build to succeed.

## 5) Live endpoint verification on Space (required)

Replace `YOUR_SPACE_URL` with your actual URL.

- [ ] Health:

```bash
curl https://YOUR_SPACE_URL/
```

- [ ] Reset:

```bash
curl -X POST https://YOUR_SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"support-easy-001"}'
```

- [ ] Tasks:

```bash
curl https://YOUR_SPACE_URL/tasks
```

- [ ] Grader:

```bash
curl https://YOUR_SPACE_URL/grader
```

- [ ] Baseline:

```bash
curl -X POST https://YOUR_SPACE_URL/baseline \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini"}'
```

## 6) Problem-statement coverage audit (required)

- [ ] Real-world simulation domain present (customer support triage).
- [ ] OpenEnv typed models + `step/reset/state` implemented.
- [ ] `openenv.yaml` present and valid.
- [ ] 3+ tasks with deterministic graders, difficulty progression easy→medium→hard.
- [ ] Reward shaping provides dense partial-progress signal + penalties.
- [ ] Baseline inference script present and reproducible.
- [ ] Docker build/run works.
- [ ] HF Space responds and `/reset` works.
- [ ] Endpoints exist: `/baseline`, `/grader`, `/tasks`.
- [ ] README includes description, spaces, tasks, setup, and baseline score documentation.

## 7) Final artifacts to submit

- [ ] Repository URL
- [ ] Hugging Face Space URL
- [ ] Latest precheck output (copy/paste log)
- [ ] Baseline JSON output (local or Space)
