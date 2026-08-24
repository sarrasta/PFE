# Tunisie Telecom — Plateforme Analytique de Rétention Client

An internal enterprise platform that turns an existing customer-retention research
project (`ml pfe/`) into a production-style application: a React dashboard, a Flask
REST API, and the original data-science pipeline running live against a real
PostgreSQL data warehouse — all Dockerized.

> Built around **real, verified research**. Every number this app shows is computed
> by the same models and formulas documented in `ml pfe/RAPPORT_PFE_FINAL.md`
> (see [ML components](#5-ml-components) and [Adaptations](#adaptations-vs-the-original-notebooks)
> for the handful of deliberate, documented changes made while integrating it).

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technologies](#3-technologies)
4. [ML components](#4-ml-components)
5. [Frontend](#5-frontend)
6. [Backend](#6-backend)
7. [Authentication and roles](#7-authentication-and-roles)
8. [Environment variables](#8-environment-variables)
9. [Docker requirements](#9-docker-requirements)
10. [Build instructions](#10-build-instructions)
11. [Run instructions](#11-run-instructions)
12. [Access URLs](#12-access-urls)
13. [API documentation](#13-api-documentation)
14. [Adding the 4 users and admin later](#14-adding-the-4-users-and-admin-later)
15. [GPU requirements](#15-gpu-requirements)
16. [Troubleshooting](#16-troubleshooting)
17. [Known limitations](#17-known-limitations)
18. [Adaptations vs. the original notebooks](#adaptations-vs-the-original-notebooks)

---

## 1. Project overview

`ml pfe/` is a completed final-year research project (a "PFE") that built 9
machine-learning objectives for telecom customer retention — churn prediction,
risk scoring, behavioral segmentation, retention targeting, ARPU forecasting,
offer-response modeling, and expected-gain scoring — against a PostgreSQL data
warehouse of 10,000 customers over 12 months (`DW_TT`). The work exists as
Jupyter notebooks and `build_*.py` generator scripts, evaluated interactively,
with results documented in `ml pfe/RAPPORT_PFE_FINAL.md`.

This repository wraps that research in a real application:

- A **Flask backend** that reproduces the exact queries, feature engineering
  and model hyperparameters from the notebooks as importable Python services,
  trains them once against the live database, and serves the results as a
  JSON API.
- A **React frontend** with a Tunisie Telecom–inspired design, giving every
  objective a dedicated page instead of a static notebook output.
- **Role-based multi-user access** (1 admin + 4 standard users), with the
  actual accounts left as environment-variable placeholders until provided.
- A **Docker Compose stack** (Postgres + backend + frontend) that restores the
  original warehouse dump automatically, so a clean machine only needs Docker.

## 2. Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌───────────────────────┐
│  React (nginx)   │ ───▶ │   Flask REST API  │ ───▶ │   ML pipeline (app/ml) │
│  frontend:8080    │ /api │   backend:5000    │      │  9 objectives, trained │
└─────────────────┘      └──────────────────┘      │  once, cached in memory│
                                    │                └───────────┬───────────┘
                                    │ SQLAlchemy                 │
                                    ▼                             ▼
                          ┌───────────────────────────────────────────┐
                          │        PostgreSQL — DW_TT (db:5432)         │
                          │  restored from ml pfe/datasql/todhia.backup │
                          │  on first container start                   │
                          └───────────────────────────────────────────┘
```

The React app never touches the database, a model file, or the filesystem —
every request goes through the Flask API, which is the only thing that talks
to Postgres and to the trained models.

**Startup lifecycle**: on boot, the backend spawns a background thread that
loads data from `DW_TT`, retrains all 9 objectives (see below), and caches the
scored 10,000-client table and every metric in memory. `/api/health` responds
immediately regardless; every other endpoint returns **HTTP 503** with
`{"error": "ml_pipeline_warming_up"}` until that first run completes — the
frontend shows a calm "training in progress" state and polls automatically.
A full run currently takes **roughly 4–6 minutes** on a modest machine
(dominated by Objectives 7 and 8's temporal feature engineering over the
120,000-row performance table); see [Known limitations](#17-known-limitations).
An admin can re-trigger it anytime from **Monitoring → Réentraîner les
modèles**, or `POST /api/admin/refresh`.

## 3. Technologies

| Layer | Stack |
|---|---|
| Frontend | React 18, React Router 6, Recharts, Vite, plain CSS (design tokens) |
| Backend | Flask 3, Gunicorn, PyJWT, SQLAlchemy |
| ML | scikit-learn, imbalanced-learn (SMOTE), pandas, numpy, scipy |
| Database | PostgreSQL 17 |
| Auth | Stateless JWT (HS256), env-configured user directory |
| Deployment | Docker, Docker Compose, nginx (static frontend + reverse proxy) |

## 4. ML components

All 9 objectives from `ml pfe/RAPPORT_PFE_FINAL.md`, reimplemented as clean
Python services under `backend/app/ml/` and orchestrated by
`backend/app/ml/pipeline.py`:

| # | Module | Objective | Model(s) |
|---|---|---|---|
| 1 | `churn_model.py` | Churn classification | SVC-RBF (C=10, γ=0.001) + SMOTE |
| 2 | `risk_model.py` | Risk level (Faible/Moyen/Eleve) | K-Means (k=3) → Random Forest |
| 3 | `churn_model.py` (`evaluate_business_metrics`) | Business-metric comparison | LR / SVC / RF / GB, cost-benefit |
| 4 | `segmentation_model.py` | Behavioral segmentation (v3, canonical) | K-Means (k=4) on composite scores → RF |
| 5 | `retention_profiles.py` | Retention targeting matrix | Segment × Risk crosstab, ROI formula |
| 6 | `propensity_model.py` | Continuous churn propensity | Ridge + isotonic calibration |
| 7 | `arpu_model.py` | Future ARPU / bill regression | RandomForest (temporal split) |
| 8 | `response_model.py` | Offer-response probability | Logistic Regression, calibrated |
| 9 | `gain_model.py` | Expected retention gain | Analytical formula (Obj. 6 × 7 × 8) |

Each module's docstring cites the exact source notebook it was ported from.
No model weights are shipped or loaded from disk — the original project never
persisted trained models either (every notebook retrains from scratch), so
the backend does the same thing at startup instead of on every request (see
[Performance](#17-known-limitations)).

**Database**: the app never modifies the source dataset; it treats
`ml pfe/datasql/todhia.backup` as a read-only warehouse snapshot, bind-mounted
into the `db` container (never copied).

## 5. Frontend

`frontend/` is a Vite + React single-page app with a sidebar/topbar shell and
one page per real capability — nothing here is a placeholder chart:

- **Dashboard** — portfolio KPIs, risk/segment distributions, urgency buckets.
- **Clients** — searchable, filterable, paginated table of all 10,000 scored
  clients; click through to a full multi-objective client profile.
- **Churn & Risque** — Objectives 1, 2, 6: distributions, profile-by-risk,
  propensity-by-segment, top-priority clients.
- **Segmentation** — Objective 4: cluster quality metrics, segment profiles.
- **Rétention** — Objectives 5, 8, 9: the 12-cell targeting matrix, lift
  curve, an interactive gain-scenario simulator, top clients by expected gain.
- **Revenus & ARPU** — Objective 7: predicted vs. actual ARPU, biggest
  predicted movers.
- **Performance des modèles** — every objective's live cross-validation
  metrics, with a raw-JSON expandable detail view.
- **Monitoring** (admin) — pipeline/service health, warehouse row counts, a
  manual re-train trigger.
- **Paramètres** (admin) — role/permission matrix, configured accounts,
  ML formula parameters.

Design: a Tunisie Telecom–inspired red/charcoal/white identity for UI chrome
(sidebar, buttons, brand mark), paired with a validated, colorblind-safe
categorical/status/sequential palette for chart data-ink — status colors
(good/warning/critical) are reserved for risk levels and never reused as a
generic series color. Every async view has explicit loading, empty, error and
"pipeline warming up" states.

## 6. Backend

```
backend/
├── app/
│   ├── __init__.py        # app factory, error handlers, pipeline bootstrap
│   ├── config.py           # env-driven config + role→scope permission map
│   ├── extensions.py       # process-wide singletons (user store, ML pipeline)
│   ├── auth/                # JWT issuing/verification, user directory, decorators
│   ├── ml/                  # the 9 objectives + data_loader.py + pipeline.py
│   ├── routes/               # one blueprint per nav section
│   └── utils/                 # shared response helpers
├── requirements.txt
├── run.py
└── Dockerfile
```

Models are trained once at startup (and on manual/scheduled refresh), cached
in memory as one `MLPipeline` instance, and read by every route — no
retraining or repeated DB round-trips per request. Inputs to `/api/clients`
filters and `/api/retention/simulate-scenario` are validated and clamped
server-side.

## 7. Authentication and roles

Auth is a small, provider-agnostic abstraction (`app/auth/`) so it can absorb
whichever real identifiers are provided later without a rewrite:

- **Username + password** — `POST /api/auth/login`
- **Personal access link/token** — `POST /api/auth/link-login` (for a
  `/access/<token>`-style distributed link instead of a password)

Both read from the same env-configured directory (see
[§14](#14-adding-the-4-users-and-admin-later)); either, both, or neither can be
set per account. Sessions are stateless JWTs (`Authorization: Bearer <token>`,
8h default expiry, `JWT_EXPIRES_MINUTES`).

**Roles**: `admin` (implicit full access) and `user` (configurable). The
scopes a role can reach are plain data in `Config.ROLE_PERMISSIONS`
(`backend/app/config.py`) — edit that dict to change what standard users see,
no route code changes needed. The backend enforces this on every request
(`@scope_required` / `@admin_required`); the frontend sidebar and routes only
*reflect* what `/api/auth/me` reports, they never decide access on their own.

## 8. Environment variables

See [`.env.example`](.env.example) for the complete, commented list. Summary:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | JWT signing secret — **required**, generate your own |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Warehouse credentials |
| `BACKEND_PORT` / `FRONTEND_PORT` / `POSTGRES_EXPOSED_PORT` | Host port mapping |
| `CORS_ORIGINS` | Only relevant if the frontend runs somewhere other than the bundled nginx |
| `JWT_EXPIRES_MINUTES` | Session length |
| `ML_REFRESH_INTERVAL_MINUTES` | 0 = manual only; >0 also retrains on a timer |
| `RETENTION_LTV_HORIZON_MONTHS` / `RETENTION_OFFER_COST_TND` | Objective 9 formula defaults |
| `ADMIN_USER` / `ADMIN_PASSWORD` / `ADMIN_ACCESS_TOKEN` / `ADMIN_NAME` | Admin account |
| `USER_1..USER_4` (+ `_PASSWORD` / `_ACCESS_TOKEN` / `_NAME` / `_ROLE`) | Standard-user accounts |
| `VITE_API_BASE_URL` | Frontend build-time API base (leave as `/api`) |

Never commit `.env` — it's git-ignored. `.env.example` contains only
placeholders (`<TO_BE_PROVIDED>` / `<TO_BE_GENERATED>`), never real secrets.

## 9. Docker requirements

Only **Docker Engine + Docker Compose v2** are required on the host — no
Python, Node, or system ML libraries. The images are CPU-only (see
[§15](#15-gpu-requirements)); the `db` image must be **PostgreSQL ≥ 17** (the
warehouse dump's archive format needs a matching-or-newer `pg_restore`; an
older server rejects it with "unsupported version in file header" — the
bundled `docker-compose.yml` already pins `postgres:17-alpine`, no action
needed unless you change that image).

## 10. Build instructions

```bash
git clone <this-repo-url>
cd PFE
cp .env.example .env
# Edit .env: set SECRET_KEY, POSTGRES_PASSWORD, and (optionally for now)
# ADMIN_USER/ADMIN_PASSWORD + USER_1.. — see §14.

docker compose build
```

## 11. Run instructions

```bash
docker compose up -d
docker compose logs -f backend   # optional: watch the first ML training run
```

First boot: Postgres restores the ~3.6MB warehouse dump (a few seconds), then
the backend trains all 9 objectives against it (a few minutes — see
[§17](#17-known-limitations)). `docker compose ps` shows all three containers
healthy once ready; `/api/health` reports `ml_pipeline.status: "ready"`.

Stop with `docker compose down` (add `-v` to also drop the database volume
and force a clean re-restore next time).

## 12. Access URLs

```
Frontend:      http://localhost:8080
Backend API:   http://localhost:5000/api/health
```

(Ports follow `FRONTEND_PORT` / `BACKEND_PORT` in `.env` if changed.)

## 13. API documentation

All routes are prefixed `/api`. Protected routes require
`Authorization: Bearer <token>` and return `401`/`403` if missing/insufficient;
ML-backed routes return `503` while the pipeline is (re)training.

| Method & path | Scope | Purpose |
|---|---|---|
| `GET /health` | public | Service + pipeline status |
| `POST /auth/login` | public | `{identifier, password}` → JWT |
| `POST /auth/link-login` | public | `{token}` → JWT |
| `GET /auth/me` | any | Current user + scopes |
| `POST /auth/logout` | any | No-op (stateless JWT) |
| `GET /dashboard` | dashboard | Portfolio KPIs |
| `GET /clients` | clients | Paginated/filtered/sorted client list |
| `GET /clients/filters` | clients | Available filter values |
| `GET /clients/<id>` | clients | Full multi-objective client profile |
| `GET /churn/overview` | churn | Objectives 1/2/6 aggregates |
| `GET /churn/top-risk` | churn | Highest-propensity clients |
| `GET /segmentation/overview` | segmentation | Objective 4 aggregates |
| `GET /retention/targeting-matrix` | retention | Objective 5 matrix |
| `GET /retention/gain-overview` | retention | Objectives 8/9 aggregates |
| `POST /retention/simulate-scenario` | retention | Re-run Obj. 9 with custom LTV/cost |
| `GET /revenue/overview` | revenue | Objective 7 aggregates |
| `GET /models` / `GET /models/<id>` | models | Live CV metrics, all objectives |
| `GET /monitoring` | monitoring (admin) | Service/pipeline/DB status |
| `GET /settings` | settings (admin) | Roles, ML params, account directory |
| `POST /admin/refresh` | admin only | Trigger a pipeline retrain |
| `GET /admin/users` | admin only | Account directory (no secrets) |

## 14. Adding the 4 users and admin later

Nothing about this app blocks on real identifiers — every account is simply
absent until configured. When you have the real values:

1. Edit `.env` (not `.env.example`) and fill in, for each of
   `ADMIN` / `USER_1` / `USER_2` / `USER_3` / `USER_4`:
   - the identifier (`ADMIN_USER`, `USER_1`, ...)
   - **either** a password (`..._PASSWORD`) **or** an access token
     (`..._ACCESS_TOKEN`), or both
   - optionally a display name (`..._NAME`) and, for standard users, a role
     override (`USER_n_ROLE`, defaults to `user`)
2. `docker compose up -d --no-deps --build backend` (or just `docker compose
   restart backend` if you only changed `.env`, since Compose re-reads it).
3. Confirm via **Paramètres → Comptes configurés** (admin) or
   `GET /api/admin/users`.

No code changes, migrations, or rebuilds of the frontend are needed.

## 15. GPU requirements

**None.** Every deployed model (SVC, RandomForest, K-Means, Ridge,
LogisticRegression, GradientBoosting) is a classic CPU-bound scikit-learn
estimator trained on a few hundred to ~10,000 rows — none of it benefits from
a GPU. The original notebooks additionally used `xgboost`, `lightgbm` and
`sdv` (CTGAN/TVAE/GaussianCopula) during Objective 1's model-comparison
iterations (v1–v4) and are not part of the final chosen models, so those
heavier, GPU-adjacent dependencies were intentionally left out of
`requirements.txt` to keep the backend image lean.

## 16. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Frontend shows "Entraînement des modèles en cours" indefinitely | Check `docker compose logs backend` for a stack trace; `/api/health` will show `ml_pipeline.status: "error"` with a message if training failed. |
| `POSTGRES_PASSWORD must be set` on `docker compose up` | `.env` is missing or wasn't copied from `.env.example`. |
| `pg_restore: error: unsupported version in file header` | The `db` image is older than PostgreSQL 17 — don't change the pinned tag in `docker-compose.yml`. |
| Login returns "Aucun compte n'est encore configuré" | No `ADMIN_USER`/`USER_n` values set yet — see [§14](#14-adding-the-4-users-and-admin-later). |
| 401 immediately after logging in | Check the system clock on the backend host — JWT `exp`/`iat` validation is time-sensitive. |
| Backend container unhealthy right after `up` | Expected briefly — it's waiting on Postgres's own healthcheck (`depends_on: condition: service_healthy`); give it ~15-20s. |
| Charts look empty right after startup | The pipeline is still training (503s from ML routes) — the topbar pill and per-page "warming up" state reflect this; no action needed, it resolves itself. |
| Want to inspect the warehouse directly | `psql -h localhost -p ${POSTGRES_EXPOSED_PORT:-5432} -U postgres -d DW_TT` from the host, or `docker compose exec db psql -U postgres -d DW_TT`. |

## 17. Known limitations

- **Cold-start time.** A full pipeline run currently takes roughly 4–6
  minutes (mostly Objectives 7/8's per-column temporal feature engineering
  over 120,000 monthly rows, each fit with cross-validated model comparison —
  faithfully reproducing the original notebooks' exact hyperparameters rather
  than trimming them for speed). This only happens at startup and on manual
  refresh, never per request.
- **Single in-process cache.** The backend intentionally runs as one Gunicorn
  worker (see `backend/Dockerfile`) because the trained models and scored
  client table live in that process's memory; scaling to more concurrent
  users would need a shared cache (e.g. Redis) in front of the pipeline
  first.
- **Historical figures vs. live figures.** The exact numbers in
  `RAPPORT_PFE_FINAL.md` were produced by a specific one-time notebook run;
  this app recomputes everything fresh from the same code and data, so live
  figures track the report closely (same fixed `random_state=42`
  throughout) but aren't guaranteed to be bit-identical across scikit-learn
  versions or if the warehouse data changes.
- **Objective 8's labels are synthetic.** The warehouse has no historical
  record of retention-offer responses, so (exactly as in the original
  research) those labels are generated from documented business rules
  calibrated to a realistic 30% response rate — this is disclosed in the UI
  and in `response_model.py`'s docstring, not hidden.
- **No production identity provider.** Auth is intentionally a minimal,
  self-contained JWT/env-directory system suited to 5 internal accounts, not
  SSO/OAuth — swap `app/auth/` for a real provider if this ever needs to
  scale beyond that.

## Adaptations vs. the original notebooks

Documented here for transparency — the underlying methodology, formulas and
hyperparameters are unchanged; these are integration-level decisions made
possible (or necessary) by turning 9 standalone notebooks into one running
system:

1. **One canonical segmentation.** The original Objective 5 notebook
   re-derived its *own* from-scratch K-Means segmentation and risk clustering
   (yielding a third, differently-named set of clusters) purely to build its
   targeting matrix. This app reuses the already-computed Objective 4
   segmentation and Objective 2 risk level everywhere, so the whole dashboard
   shares one consistent taxonomy. The matrix's financial formulas and
   campaign-assignment logic are unchanged — only the labels feeding them are
   now shared. See `retention_profiles.py`.
2. **Objective 9 uses real upstream scores, not inline proxies.** The
   original `build_09.py` could only run standalone, so it re-derived its own
   proxy formulas for "P_churn" and "ARPU_predit" instead of using
   Objective 6's and Objective 7's actual outputs — the report itself lists
   wiring those in as a "next step." Since this app already keeps all 9
   objectives in memory together, `gain_model.py` uses the real Objective 6
   propensity score and Objective 7 ARPU prediction directly.
3. **A fixed feature-ordering bug found during integration.** Porting
   Objective 6, we found that scoring the full 10,000-client base reused a
   feature list positionally derived from a differently-ordered SQL query
   than the one used to train on the 422 labeled clients — silently swapping
   two columns and corrupting every deployed propensity score. Fixed by
   capturing the exact fitted feature order once and reusing it by name at
   scoring time (`propensity_model.py::_engineer`); verified against the
   report's reference statistics (mean 0.684 live vs. 0.691 reported) after
   the fix.
