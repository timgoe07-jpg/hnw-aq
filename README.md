# Capspace Risk & Engagement Prototype (Flask + Angular)

Small end-to-end prototype that seeds synthetic investor/loan data, trains simple ML models (scikit-learn), serves predictions + a Daily Risk & Engagement report via Flask, and visualises everything in an Angular Material dashboard.

## Prerequisites
- Python 3.10+
- Node 18+ and npm

## Backend (Flask + SQLite)
```bash
cd backend

# create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# seed synthetic data into backend/app.db
python seed_data.py

# train and persist churn + default models (writes to backend/models/*.joblib)
python train_models.py

# run the API
python app.py  # serves http://localhost:5000
```
Notes:
- CORS defaults to `http://localhost:4200`; override with `CORS_ORIGINS` env var if needed.
- Database lives at `backend/app.db`.

## Frontend (Angular + Angular Material)
```bash
cd frontend
npm install
npm start   # ng serve on http://localhost:4200
```
Routes:
- `/dashboard` - bucket counts and top at-risk investors/loans plus a report preview.
- `/predictions` - forms to run investor churn and loan default predictions.
- `/report` - full Daily Risk & Engagement report with copy-to-clipboard.

## Key API endpoints (all under http://localhost:5000/api)
- `GET /health` - health check.
- `GET /investors` / `GET /loans` - paginated listings with latest scores.
- `POST /predict/investor_churn` - predict churn from features or `investor_id`.
- `POST /predict/loan_default` - predict default from features or `loan_id`.
- `POST /predict/batch_refresh` - recompute and store scores for all records.
- `POST /report/daily` - generate summary KPIs + Markdown report.

## Run order
1) `python seed_data.py`  
2) `python train_models.py`  
3) `python app.py`  
4) `npm start` (in `frontend`) and open `http://localhost:4200`.

All data is synthetic; no external APIs are used.
