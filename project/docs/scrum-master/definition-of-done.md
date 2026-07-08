# Definition of Done — ACE Bootcamp Team 10 (CKD Risk Prediction)

A task or user story is considered **Done** only when all of the following are true:

## Code

- [ ] Code is written and functions as intended
- [ ] Code follows the project's naming and structure conventions
- [ ] No hardcoded values that should be configurable (paths, thresholds, credentials, etc.)
- [ ] Code is modular, readable, and follows PEP 8 standards
- [ ] Proper error handling and input validation are implemented
- [ ] Code has been peer-reviewed by at least one other team member (via Pull Request)

## Testing

- [ ] Feature has been manually tested locally
- [ ] Relevant unit tests written and passing (where applicable)
- [ ] Edge cases and invalid inputs have been tested
- [ ] No known bugs introduced by this change
- [ ] Existing functionality verified to prevent regressions

## Model-specific (for ML tasks)

- [ ] Model trained and evaluated on the CKD dataset with a fixed random seed for reproducibility
- [ ] Data preprocessing pipeline verified and applied consistently
- [ ] Key metrics (Accuracy, Precision, Recall, F1-Score) recorded
- [ ] Recall on the CKD-positive class meets the agreed project target and is treated as the primary evaluation metric
- [ ] Model checked for overfitting and data leakage
- [ ] Feature importance or model explainability validated (where applicable)
- [ ] Model artifact saved (e.g., via `joblib`) and referenced correctly in `app.py`

## Documentation

- [ ] README updated if setup, installation, or usage steps changed
- [ ] Docstrings/comments added for non-obvious logic
- [ ] Project documentation updated where applicable
- [ ] Relevant ClickUp task updated with implementation notes and status

## Integration

- [ ] Changes merged into `main` via Pull Request (not pushed directly, where team convention requires PRs)
- [ ] No merge conflicts left unresolved
- [ ] Application runs end-to-end locally after the change (`streamlit run app.py` works without errors)
- [ ] Model loads successfully within the application
- [ ] App displays a disclaimer that predictions are not a medical diagnosis

## Sprint Process

- [ ] Acceptance Criteria fully satisfied
- [ ] Task moved to the **Done** column in ClickUp
- [ ] Task discussed and confirmed during Daily Scrum or Sprint Review
- [ ] Sprint Goal remains achievable after completing the task
