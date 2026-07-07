# Definition of Done — ACE Bootcamp Team 10 (CKD Risk Prediction)

A task or user story is considered **Done** only when all of the following are true:

## Code
- [ ] Code is written and functions as intended
- [ ] Code follows the project's naming and structure conventions
- [ ] No hardcoded values that should be configurable (paths, thresholds, etc.)
- [ ] Code has been peer-reviewed by at least one other team member (via Pull Request)

## Testing
- [ ] Feature has been manually tested locally
- [ ] Relevant unit tests written and passing (where applicable)
- [ ] No known bugs introduced by this change

## Model-specific (for ML tasks)
- [ ] Model trained and evaluated on the CKD dataset
- [ ] Key metrics (accuracy, precision, recall, F1) recorded
- [ ] Model artifact saved (e.g. via joblib) and referenced correctly in `app.py`

## Documentation
- [ ] README updated if setup/usage steps changed
- [ ] Docstrings/comments added for non-obvious logic
- [ ] Relevant ClickUp task updated with status and notes

## Integration
- [ ] Changes merged into `main` via Pull Request (not pushed directly, where team convention requires PRs)
- [ ] No merge conflicts left unresolved
- [ ] App runs end-to-end locally after the change (`streamlit run app.py` works without errors)

## Sprint Process
- [ ] Task moved to "Done" column in ClickUp
- [ ] Task discussed/confirmed in standup or sprint review

---
*This document should be reviewed and updated by the team at each retrospective if the team's working definition evolves.*
