# Submissions

Create `submissions/YOUR_TEAM_NAME/` with:

```
submission.enc
meta.json
```

Encrypt your prediction ZIP:

```bash
python -m encryption.encrypt predictions.zip submissions/YOUR_TEAM_NAME/submission.enc
```

`meta.json` example:

```json
{"team": "YOUR_TEAM_NAME", "model": "My U-Net", "type": "human", "notes": "Short description"}
```

Open a PR that changes only these two files.
