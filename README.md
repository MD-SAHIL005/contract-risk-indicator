# AI-Based Contract Risk Indicator — Public Website Deployment

This folder is prepared for deployment on Streamlit Community Cloud.

## Files
- `app.py`: Streamlit web app
- `contract_risk_model.joblib`: trained clause-category model
- `contract_clauses.csv`: training examples (kept for project transparency)
- `requirements.txt`: pinned core ML dependencies
- `sample_contract_for_risk_indicator.pdf`: fictional test contract

## Deploy it online
1. Create/sign in to a GitHub account.
2. Create a new public repository, e.g. `contract-risk-indicator`.
3. Upload the files in this folder to the repository root. Do not upload `.venv` folders.
4. Open https://share.streamlit.io/ and connect GitHub.
5. Choose the repository, branch `main`, and main file path `app.py`.
6. Click Deploy. Wait for dependency installation and startup.
7. Copy the Streamlit app URL and share it with anyone.

## Important notes
- A public repository makes the source code and included model publicly downloadable.
- Uploaded contract documents are processed by the hosted app. Do not upload confidential or personal contracts unless you understand the hosting provider's privacy and retention terms.
- This educational prototype classifies clause categories, not validated legal risk. Model confidence is not the probability that a clause is legally risky. Human review is required.
- If Streamlit Cloud reports a package/build error, review the build logs before changing package versions.
