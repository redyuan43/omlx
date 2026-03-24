# AMD AGX Parity User Units

These user-level `systemd` unit files mirror the AMD AGX parity topology without
writing directly into `~/.config/systemd/user`.

Copy or symlink these files into:

- `~/.config/systemd/user/omlx-agx-main.service`
- `~/.config/systemd/user/omlx-agx-ocr.service`

Then reload and enable them:

```bash
systemctl --user daemon-reload
systemctl --user enable --now omlx-agx-main.service
systemctl --user enable --now omlx-agx-ocr.service
```

The main service registers `embed-text` and `rerank-qwen` after startup.
