# Next Problem: User Manual

Nov 18, 2026 · Venkata Sreekar

Next Problem suggests Codeforces problems that are hard enough to teach you something but not so hard that they frustrate you. Enter your Codeforces handle and you get problems you have a 50-70% chance of solving, focused on your weakest topics.

This manual has two parts. **Part 1 (sections 1 to 4)** is for anyone who wants practice problems and needs no technical knowledge. **Part 2 (sections 5 to 8)** is for the person who installs and runs the system.

# Part 1: Using Next Problem

## 1. Opening the app

Open a web browser (Chrome, Firefox, Edge or Safari) and go to **http://localhost:8080** on the computer where Next Problem runs. There is no account to create, no password and nothing to install.

You need two things:

- **Your Codeforces handle,** the user name shown on your Codeforces profile.
- **At least one rated contest** on that account. The app learns your strengths and weaknesses from your contest history, so it cannot help an account that has never taken part in a rated contest.

If the page does not open, the system is not running. Ask the person who runs it to start it (section 5).

## 2. Getting recommendations

Type your handle, choose how many problems you want, and press **Get problems**. The first search for a handle takes about 10 seconds; searches after that are almost instant.

1. Open http://localhost:8080.
2. In the text box, type your Codeforces handle, the name shown on your Codeforces profile (for example `Venkata_Sreekar`).
3. In the drop-down list, choose **5**, **10** (the default) or **20** problems.
4. Press **Get problems**. While it works, the page shows "Fetching your data from Codeforces. This can take about 10 seconds the first time." and the button is greyed out.
5. Read the results (section 3). Click a problem's name to open it on Codeforces in a new tab.
6. Solve a few problems, then come back later for a new set. Problems you have solved, in a contest or in practice, are never recommended again.

The page also has a **How to use** section near the top. Click it to show a short version of these steps.

**Tips:**

- New solves show up in your recommendations within about 10 minutes, because the app keeps your Codeforces data for 10 minutes before fetching it again.
- The page always uses the 50-70% range. To try a different range, call the API directly, for example `http://localhost:8000/recommend?handle=Venkata_Sreekar&low=0.4&high=0.6`. The answer comes back as raw JSON.
- The page works on phones and narrow windows: the table scrolls sideways if it does not fit.

## 3. Reading the results

The results have three parts: a summary line, your weakest topics, and a table of problems. The problems at the top of the table target your weakest topics.

**Summary line,** for example *"\<handle> (rating \<your rating>): \<n> suitable problems found, showing the top 10."* It gives your current rating and how many unsolved problems fell in the 50-70% range. The table shows the best of them.

**Your weakest topics:** up to three topic tags where your success rate in recent rated contests is lowest. Only topics you have met at least 5 times count, so a single bad attempt does not make a topic "weak".

**The table:**

| Column | Meaning |
| --- | --- |
| # | Position in the list. Number 1 is the best match. |
| Problem | Contest number, problem letter and name, for example *\<contest>\<letter> - \<problem name>*. Click it to open the problem on Codeforces. |
| Rating | The problem's difficulty rating on Codeforces |
| Topics | The problem's Codeforces tags |
| Solve chance | The model's estimate that you would solve this problem, between 50% and 70% |

**How the list is ordered:** first by how weak you are in the problem's topics, then by how close its solve chance is to 60%. Problems without any tags therefore appear last.

**What "solve chance" means:** the model learned from rated contests, so it estimates your chance of solving the problem under contest conditions. In relaxed practice your real chance is usually somewhat higher. Treat the range as "challenging but reachable".

**If the table is empty,** the page says *"No problems in the 50-70% range were found for this handle."* This can happen to very high- or very low-rated users. Try the API with a wider range (section 2, Tips).

## 4. Messages and what to do

Every problem shows a message in a coloured box above the results. The table lists each message, its cause and the fix.

| Message | What it means | What to do |
| --- | --- | --- |
| Please enter a Codeforces handle. | The box was empty | Type your handle and press Get problems again |
| Invalid handle: use 3-24 letters, digits, '\_', '-' or '.'. | What you typed cannot be a Codeforces handle (the page checks this before searching) | Copy the handle exactly from your Codeforces profile |
| Handle '...' was not found on Codeforces. | No Codeforces account has this name | Check the spelling |
| This user has no rating yet. Play a rated contest first. | The account has never taken part in a rated contest | Take part in a rated contest, then try again |
| Codeforces is not reachable right now. Try again later. | Codeforces is busy or down; this is not a problem with your handle | Wait a minute and try again |
| The model service is unavailable. Try again later. | Part of Next Problem has stopped | Try again in a minute; if it repeats, tell the person who runs the system (section 8) |
| Cannot reach the recommendation service. Please check that it is running. | Next Problem is not running | Tell the person who runs the system (section 8) |
| Something went wrong (error ...). | An unexpected error | Try again; if it repeats, tell the person who runs the system |
| No problems in the 50-70% range were found for this handle. | Not an error: no unsolved problem fitted the range | See section 3 |

# Part 2: For the person who runs the system

Sections 5 to 8 need a terminal and some technical knowledge. If you only want practice problems, you can stop at section 4.

## 5. Setting up and starting the system

One command starts the whole system: `docker compose up -d`. Then open http://localhost:8080 in a browser.

### 5.1 What you need

- **To use the app:** Docker with Docker Compose, a browser, and an internet connection (recommendations use live Codeforces data).
- **To rebuild the data or the model as well:** Git, DVC and conda.
- **A rated Codeforces account,** meaning you have taken part in at least one rated contest. The app cannot recommend for unrated users.

### 5.2 First-time setup

Run these from a terminal:

1. Get the code and enter the folder:

   ```bash
   git clone <repository URL> next-problem
   cd next-problem
   ```
2. Create the Python environment (needed for DVC and the pipeline):

   ```bash
   conda env create -f conda.yaml
   conda activate nextproblem-env
   ```
3. Download the data and the trained model from the DVC remote:

   ```bash
   dvc pull
   ```

   If the remote is not reachable, rebuild everything with `dvc repro` instead. That downloads 70 contests from Codeforces and retrains, so it takes much longer.
4. Build and start the five services:

   ```bash
   docker compose up -d --build
   ```
5. Check that everything is running:

   ```bash
   docker compose ps
   ```

   All five services (`frontend`, `backend`, `model`, `prometheus`, `grafana`) should show **Up**. The backend waits until the model server is healthy, so give it about 30 seconds.

### 5.3 Addresses

| What | Address |
| --- | --- |
| **The app** | http://localhost:8080 |
| Pipeline status page | http://localhost:8080/pipeline.html |
| API documentation | http://localhost:8000/docs |
| Grafana dashboard (login `admin` / `admin`) | http://localhost:3001 |
| Prometheus | http://localhost:9090 |
| MLflow (start it first, see section 6.2) | http://localhost:5000 |

After the first setup, you only need `docker compose up -d` from the project folder.

## 6. Pipeline status page and monitoring tools

These pages are for whoever runs the system: they show how the model was built, how good it is, and how the live service behaves. The app's footer and the Pipeline status page link to all of them.

### 6.1 Pipeline status page

Open it with the **Pipeline status** link at the bottom of the app, or at http://localhost:8080/pipeline.html. **Back to the app** returns to the main page.

| Part | Shows |
| --- | --- |
| Services | Whether the backend and the model server are up |
| Pipeline stages | The six stages (Fetch, Features, Train, Register, Serve, Monitor), the tool behind each, and their latest numbers, such as rows processed and how many data quality checks passed |
| Model results | Test and validation log loss and AUC for the Elo baseline and the model, and whether the model was accepted |
| Tools | Links to MLflow, Grafana, Prometheus and the API documentation |

The numbers come from the last time the backend image was built. After retraining, rebuild it (section 7).

### 6.2 MLflow: experiments and models

MLflow is not a Docker service, so start it from the project folder with the conda environment active, and leave the terminal open:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open http://localhost:5000:

- **Experiments → next-problem:** each training run is a `train_pipeline` row. Expand it to see `elo_baseline` and `hgb_0` to `hgb_2`, with their parameters, metrics and files.
- **Models → next-problem-classifier:** every registered version. The one marked `champion` is the model being served.

### 6.3 Grafana: monitoring dashboard

1. Open http://localhost:3001 and log in with `admin` / `admin`. Skip the password-change prompt.
2. Go to **Dashboards → Next Problem - Monitoring**, or open http://localhost:3001/d/next-problem directly.
3. The panels show whether the backend is up, request counts, latency, errors by type and feature drift. At the bottom, *Service status* shows every component (backend, model server, frontend, Prometheus, Grafana) as UP or DOWN, and *Alerts firing* and *Firing alerts* show any active alerts.
4. If panels are empty, make a few searches in the app, wait 30 seconds, and widen the time range (top right) to *Last 1 hour*.

### 6.4 Prometheus

- http://localhost:9090/targets shows whether Prometheus can reach the backend, Grafana and itself (state **UP**).
- http://localhost:9090/alerts lists the 7 alert rules. An alert turns **Pending** when its condition starts and **Firing** once it has lasted long enough, for example `ModelServerDown` after 1 minute.
- On http://localhost:9090/graph, enter a query such as `sum by (endpoint, status) (np_requests_total)` and press Execute to count requests per endpoint and outcome.

### 6.5 API documentation

http://localhost:8000/docs lists every endpoint with its parameters and responses. Open `/recommend`, click **Try it out**, enter a handle and press **Execute** to call the API from the browser.

## 7. Stopping, restarting and updating

Use `docker compose stop` to pause and `docker compose start` to resume. After retraining, rebuild the backend and model images so the app serves the new model. Run every command from the project folder.

| To | Run | Notes |
| --- | --- | --- |
| Pause everything | `docker compose stop` | Containers are kept |
| Resume | `docker compose start` |  |
| Shut down completely | `docker compose down` | Removes the containers but not the images, code or data. Prometheus history is lost, because it is stored inside its container. |
| Start again after `down` | `docker compose up -d` | Always without a service name, so all five services start |
| Check what is running | `docker compose ps` |  |
| Read a service's log | `docker compose logs backend` (or `model`, `grafana`, ...) | Add `-f` to follow it live |

### 7.1 After retraining the model

1. Run the pipeline. DVC reruns only the stages whose inputs changed. A new model is registered, becomes `champion` only if it beats the Elo baseline, and is exported to `model/`:

   ```bash
   dvc repro
   ```

   Or run the full CI, which also runs the tests and the acceptance gate: `scripts/ci.sh`.
2. Rebuild the two images that contain the model and its metrics:

   ```bash
   docker compose up -d --build backend model
   ```
3. Open the Pipeline status page and check that the model results show the new numbers.
4. Save the new versions:

   ```bash
   git add -A && git commit -m "Retrain model"
   dvc push
   ```

### 7.2 Changing addresses or ports

- **Links and backend address used by the page:** edit `frontend/config.js` and refresh the browser. No rebuild is needed.
- **Published ports:** edit the `ports` lines in `docker-compose.yml`, then run `docker compose up -d`. If you change the backend port, update `API_URL` in `config.js` to match.

## 8. Troubleshooting

Most problems come from a service that is not running. Start with `docker compose ps`, then `docker compose up -d`.

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `docker compose up --build` fails on `COPY model` or `COPY data/processed/...` | The model or the data files were never downloaded | Run `dvc pull` (or `dvc repro`), then build again |
| "port is already allocated" | Another program uses 8080, 8000, 5001, 9090 or 3001 | Stop that program, or change the port (section 7.2) |
| The backend never starts | It waits for the model server to become healthy | `docker compose logs model` shows why the model server fails |
| Every search says "Cannot reach the recommendation service" | Backend stopped, or a wrong `API_URL` in `config.js` | `docker compose up -d`; check `config.js`; press F12 in the browser and look for errors in the Console tab |
| The Grafana or Prometheus link does not open | Those containers are not running, often after starting only one service | `docker compose up -d` (no service name) |
| The MLflow link does not open | MLflow is not a Docker service | Start it as in section 6.2 |
| Grafana has no *Next Problem - Monitoring* dashboard | Grafana did not load its provisioning files | `docker compose up -d --force-recreate grafana`; if still missing, check `docker compose logs grafana` |
| Grafana panels are empty | No traffic yet, or the time range is too short | Make a few searches, wait 30 s, choose *Last 1 hour* |
| "Codeforces is not reachable" appears often | Codeforces limits requests (about one every 2 s) or is down | Wait a minute; check that https://codeforces.com opens |
| Solved problems still appear | Your Codeforces data is kept for 10 minutes | Wait 10 minutes and search again |
| The drift panels show large values, or the `FeatureDrift` alert fires | Expected: the app looks at up to 15 recent contests, more history than most training rows had | No action needed; see HLD sections 8.3 and 10 |
| `dvc repro` stops at the features stage with "Data quality checks FAILED" | The downloaded data broke one of the 11 checks; the log names which | Look at `quality_checks` in `data/processed/spark_metrics.json`; re-download the affected contest files and run `dvc repro` again |
| An alert such as `ModelServerDown` is firing | That service has stopped or is failing its health check | `docker compose ps`, then `docker compose up -d`; read that service's log |
