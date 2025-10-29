AIoT-hw3
========

This repository contains exercises and change proposals for an AIoT homework project. It uses OpenSpec for spec-driven change management. The `openspec/` directory holds project conventions, specs, and change proposals.

Quick links
-----------
- OpenSpec instructions: `openspec/AGENTS.md`
- Project conventions: `openspec/project.md`
- Change proposals: `openspec/changes/`

Spam classification change
--------------------------
Change: `openspec/changes/add-spam-classification/`

This change adds a spam-classification capability (baseline) including training and an interactive Streamlit demo. See `openspec/changes/add-spam-classification/README.md` for quick start and details.

How to run the spam demo locally
-------------------------------
1. Create a virtual environment, activate it, and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r openspec/changes/add-spam-classification/requirements.txt
```

2. Train the baseline model:

```bash
python3 openspec/changes/add-spam-classification/train.py
```

3. Start the Streamlit demo:

```bash
streamlit run openspec/changes/add-spam-classification/streamlit_app.py
```

Open http://localhost:8501 in your browser.

OpenSpec workflow
-----------------
- Create change proposals under `openspec/changes/<change-id>/`
- Add `proposal.md`, `tasks.md`, and spec deltas under `specs/`
- Validate changes: `openspec validate <change-id> --strict`
- Archive changes after deployment: `openspec archive <change-id>`

License & Attribution
---------------------
Check dataset and dependency licenses before redistribution or commercial usage. The SMS dataset is from PacktPublishing's example repository. See files under `openspec/changes/add-spam-classification/` for details.

Contact
-------
For follow-ups (tests, containerization, deployment), open an issue or request the next phase in the change proposal.
